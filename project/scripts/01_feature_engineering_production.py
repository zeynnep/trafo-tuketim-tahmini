import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import time

t0 = time.time()
pd.set_option('display.width', 160)

# ============================================================
# 1. LOAD
# ============================================================
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')
train['tarih'] = pd.to_datetime(train['tarih'])
test['tarih'] = pd.to_datetime(test['tarih'])
train['tanim'] = train['tanim'].astype(str)
test['tanim'] = test['tanim'].astype(str)

def split_lok(df):
    parts = df['lokasyon'].str.split('>', expand=True)
    df['il'] = parts[0]
    df['bolge'] = parts[1]
    df['ilce'] = parts[2] if parts.shape[1] > 2 else parts[1]
    df['ilce'] = df['ilce'].fillna(df['bolge'])
    return df

train = split_lok(train)
test = split_lok(test)
train['log_tuketim'] = np.log1p(train['tuketim'].clip(lower=0))
train = train.sort_values(['tanim', 'tarih']).reset_index(drop=True)
print(f"[{time.time()-t0:.1f}s] Loaded. train={train.shape} test={test.shape}")

# ============================================================
# 2. PER-ACTUAL-RECORD "SNAPSHOT" FEATURES (inclusive -> "what we'd know
#    if today were this row's date")
# ============================================================
g = train.groupby('tanim')['log_tuketim']
train['snap_exp_count'] = g.cumcount() + 1
train['snap_exp_mean'] = g.expanding().mean().reset_index(level=0, drop=True)
train['snap_exp_std'] = g.expanding().std().reset_index(level=0, drop=True)
train['snap_exp_min'] = g.expanding().min().reset_index(level=0, drop=True)
train['snap_exp_max'] = g.expanding().max().reset_index(level=0, drop=True)
train['snap_last_val'] = train['log_tuketim']

train_idx = train.set_index('tarih')
train['snap_roll30'] = train_idx.groupby('tanim')['log_tuketim'].rolling('30D').mean().reset_index(level=0, drop=True).values
train['snap_roll90'] = train_idx.groupby('tanim')['log_tuketim'].rolling('90D').mean().reset_index(level=0, drop=True).values

# trend/egim feature'i: trafonun son 30 gunu, son 90 gununden yuksek mi dusuk mu?
# pozitif -> tuketim yakin zamanda artiyor, negatif -> azaliyor (momentum sinyali)
train['snap_trend'] = train['snap_roll30'] - train['snap_roll90']

train['dow_tmp'] = train['tarih'].dt.dayofweek
train['snap_dow_mean'] = (train.groupby(['tanim', 'dow_tmp'])['log_tuketim']
                           .expanding().mean().reset_index(level=[0, 1], drop=True)).values

# per-tanim per-calendar-month seasonal mean (captures e.g. "this trafo's July spike")
# expanding across years -> for a 2026 target this becomes ~"same month last year" once available
train['cal_month_tmp'] = train['tarih'].dt.month
train['snap_month_mean'] = (train.groupby(['tanim', 'cal_month_tmp'])['log_tuketim']
                             .expanding().mean().reset_index(level=[0, 1], drop=True)).values

first_seen = train.groupby('tanim')['tarih'].transform('min')
train['snap_age_days'] = (train['tarih'] - first_seen).dt.days
train['snap_date'] = train['tarih']

print(f"[{time.time()-t0:.1f}s] Snapshot features built")

snapshot_cols = ['snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
                  'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_trend', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days']
snapshot_table = train[['tanim', 'snap_date'] + snapshot_cols].sort_values(['tanim', 'snap_date']).reset_index(drop=True)

guc_edges = pd.qcut(train['guc'], 20, duplicates='drop').cat.categories
train['guc_bucket'] = pd.cut(train['guc'], bins=guc_edges)
tanim_meta = train[['tanim', 'guc', 'guc_bucket', 'il', 'bolge', 'ilce']].drop_duplicates('tanim')

FEATURE_COLS = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_trend', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
]
CAT_COLS = ['il', 'bolge', 'ilce']


def build_examples(cutoff, target_dates_df):
    cutoff = pd.Timestamp(cutoff)
    hist = train[train['tarih'] <= cutoff]

    tdf = target_dates_df.copy()
    snap_hist = snapshot_table[snapshot_table['snap_date'] <= cutoff]
    merged = pd.merge_asof(
        tdf.sort_values('tarih'), snap_hist.sort_values('snap_date'),
        left_on='tarih', right_on='snap_date', by='tanim', direction='backward'
    )

    hist_im = hist.merge(tanim_meta[['tanim', 'ilce']], on='tanim', how='left', suffixes=('', '_m'))
    hist_im['cal_month'] = hist_im['tarih'].dt.month
    ilce_month_mean = hist_im.groupby(['ilce', 'cal_month'])['log_tuketim'].mean().reset_index()
    ilce_month_mean.columns = ['ilce', 'cal_month', 'fb_ilce_month_mean']
    ilce_mean = hist_im.groupby('ilce')['log_tuketim'].mean().reset_index()
    ilce_mean.columns = ['ilce', 'fb_ilce_mean']

    hist_gb = hist.merge(tanim_meta[['tanim', 'guc_bucket', 'ilce']], on='tanim', how='left', suffixes=('', '_m'))
    guc_ilce_mean = hist_gb.groupby(['guc_bucket', 'ilce'], observed=True)['log_tuketim'].mean().reset_index()
    guc_ilce_mean.columns = ['guc_bucket', 'ilce', 'fb_guc_ilce_mean']

    # nearest-guc-neighbor within same ilce (finer than bucket averaging): for each
    # known trafo (as of cutoff) compute its own mean, then asof-match by guc within ilce
    tanim_own_mean = hist.groupby('tanim')['log_tuketim'].mean().reset_index()
    tanim_own_mean.columns = ['tanim', 'own_mean']
    knn_ref = tanim_meta[['tanim', 'guc', 'ilce']].merge(tanim_own_mean, on='tanim', how='inner')
    knn_ref = knn_ref.sort_values('guc')

    global_mean = hist['log_tuketim'].mean()

    merged = merged.merge(tanim_meta, on='tanim', how='left')
    # for tanim never seen in train at all, tanim_meta merge leaves guc/il/bolge/ilce
    # as NaN -> fall back to the row's own known attributes (always present in test.csv)
    if 'own_guc' in merged.columns:
        merged['guc'] = merged['guc'].fillna(merged['own_guc'])
        merged['il'] = merged['il'].fillna(merged['own_il'])
        merged['bolge'] = merged['bolge'].fillna(merged['own_bolge'])
        merged['ilce'] = merged['ilce'].fillna(merged['own_ilce'])
        merged['guc_bucket'] = merged['guc_bucket'].astype('object')
        need_bucket = merged['guc_bucket'].isna() & merged['guc'].notna()
        if need_bucket.any():
            merged.loc[need_bucket, 'guc_bucket'] = pd.cut(merged.loc[need_bucket, 'guc'], bins=guc_edges)
        merged = merged.drop(columns=['own_guc', 'own_il', 'own_bolge', 'own_ilce'])
    merged['cal_month'] = merged['tarih'].dt.month
    merged = merged.merge(ilce_month_mean, on=['ilce', 'cal_month'], how='left')
    merged = merged.merge(ilce_mean, on='ilce', how='left')
    merged = merged.merge(guc_ilce_mean, on=['guc_bucket', 'ilce'], how='left')

    if len(knn_ref) > 0:
        merged['guc'] = merged['guc'].astype('float64')
        knn_ref = knn_ref.copy()
        knn_ref['guc'] = knn_ref['guc'].astype('float64')
        merged = merged.sort_values('guc')
        merged = pd.merge_asof(
            merged, knn_ref[['guc', 'ilce', 'own_mean']].rename(columns={'own_mean': 'fb_knn_guc_ilce'}),
            on='guc', by='ilce', direction='nearest'
        )
    else:
        merged['fb_knn_guc_ilce'] = np.nan
    merged['fb_guc_ilce_mean'] = merged['fb_knn_guc_ilce'].fillna(merged['fb_guc_ilce_mean'])

    merged['fb_ilce_month_mean'] = merged['fb_ilce_month_mean'].fillna(merged['fb_ilce_mean']).fillna(global_mean)
    merged['fb_ilce_mean'] = merged['fb_ilce_mean'].fillna(global_mean)
    merged['fb_guc_ilce_mean'] = merged['fb_guc_ilce_mean'].fillna(merged['fb_ilce_mean']).fillna(global_mean)

    for c in ['snap_exp_mean', 'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean']:
        merged[c] = merged[c].fillna(merged['fb_guc_ilce_mean']).fillna(merged['fb_ilce_month_mean']) \
                              .fillna(merged['fb_ilce_mean']).fillna(global_mean)
    merged['snap_exp_std'] = merged['snap_exp_std'].fillna(0)
    merged['snap_trend'] = merged['snap_trend'].fillna(0)
    merged['snap_exp_min'] = merged['snap_exp_min'].fillna(merged['snap_exp_mean'])
    merged['snap_exp_max'] = merged['snap_exp_max'].fillna(merged['snap_exp_mean'])
    merged['snap_exp_count'] = merged['snap_exp_count'].fillna(0)
    merged['is_new_tanim'] = (merged['snap_exp_count'] == 0).astype(int)

    merged['recency_days'] = (cutoff - merged['snap_date']).dt.days
    merged['recency_days'] = merged['recency_days'].fillna(9999)
    merged['horizon_days'] = (merged['tarih'] - cutoff).dt.days
    merged['snap_age_days'] = merged['snap_age_days'].fillna(0)

    merged['month'] = merged['tarih'].dt.month
    merged['day'] = merged['tarih'].dt.day
    merged['dayofweek'] = merged['tarih'].dt.dayofweek
    merged['dayofyear'] = merged['tarih'].dt.dayofyear
    merged['weekofyear'] = merged['tarih'].dt.isocalendar().week.astype(int)
    merged['is_weekend'] = (merged['dayofweek'] >= 5).astype(int)
    merged['quarter'] = merged['tarih'].dt.quarter
    return merged


# ============================================================
# 3. BUILD AUGMENTED TRAINING SET (multiple cutoffs = multiple forecast origins)
#    Validation cutoff (2025-03-31) is EXCLUDED from training augmentation.
# ============================================================
train_cutoffs = ['2025-04-30', '2025-05-31', '2025-06-30', '2025-07-31', '2025-08-31',
                  '2025-09-30', '2025-10-31', '2025-11-30', '2025-12-31', '2026-01-31']
MAX_HORIZON = 123

aug_frames = []
for c in train_cutoffs:
    c = pd.Timestamp(c)
    horizon_end = c + pd.Timedelta(days=MAX_HORIZON)
    targets = train[(train['tarih'] > c) & (train['tarih'] <= horizon_end)][['tanim', 'tarih', 'log_tuketim']]
    if len(targets) == 0:
        continue
    ex = build_examples(c, targets[['tanim', 'tarih']])
    ex['target'] = targets.set_index(['tanim', 'tarih']).loc[
        list(zip(ex['tanim'], ex['tarih']))]['log_tuketim'].values
    aug_frames.append(ex)
    print(f"[{time.time()-t0:.1f}s] cutoff {c.date()}: {len(ex)} examples")

train_examples = pd.concat(aug_frames, ignore_index=True)
print(f"[{time.time()-t0:.1f}s] Total augmented training examples: {len(train_examples)}")

# ============================================================
# 4. VALIDATION SET: cutoff=2025-03-31, targets 2025-04-01..2025-07-31
#    (mirrors the real task exactly: same horizon length, same structure)
# ============================================================
val_cutoff = pd.Timestamp('2025-03-31')
val_horizon_end = val_cutoff + pd.Timedelta(days=MAX_HORIZON)
val_targets = train[(train['tarih'] > val_cutoff) & (train['tarih'] <= val_horizon_end)][['tanim', 'tarih', 'log_tuketim']]
val_examples = build_examples(val_cutoff, val_targets[['tanim', 'tarih']])
val_examples['target'] = val_targets.set_index(['tanim', 'tarih']).loc[
    list(zip(val_examples['tanim'], val_examples['tarih']))]['log_tuketim'].values
print(f"[{time.time()-t0:.1f}s] Validation examples: {len(val_examples)}")

# ============================================================
# 5. TEST SET: cutoff = last train date, targets = actual test rows
# ============================================================
test_cutoff = train['tarih'].max()
test_own = test[['tanim', 'tarih', 'guc', 'il', 'bolge', 'ilce']].rename(
    columns={'guc': 'own_guc', 'il': 'own_il', 'bolge': 'own_bolge', 'ilce': 'own_ilce'})
test_examples = build_examples(test_cutoff, test_own)
test_examples['id'] = test['id'].values if len(test_examples) == len(test) else None
# safer: merge by tanim+tarih to attach id correctly (build_examples may reorder)
test_examples = test_examples.merge(test[['tanim', 'tarih', 'id']], on=['tanim', 'tarih'], how='left')
print(f"[{time.time()-t0:.1f}s] Test examples: {len(test_examples)} (should equal {len(test)})")

for c in CAT_COLS:
    all_cats = pd.concat([train_examples[c], val_examples[c], test_examples[c]]).astype('category').cat.categories
    train_examples[c] = pd.Categorical(train_examples[c], categories=all_cats)
    val_examples[c] = pd.Categorical(val_examples[c], categories=all_cats)
    test_examples[c] = pd.Categorical(test_examples[c], categories=all_cats)

drop_extra = ['guc_bucket', 'cal_month', 'snap_date', 'dow_tmp', 'cal_month_tmp']
for df_ in (train_examples, val_examples, test_examples):
    for c in drop_extra:
        if c in df_.columns:
            df_.drop(columns=c, inplace=True)

train_examples.to_parquet('/home/claude/work/train_examples.parquet')
val_examples.to_parquet('/home/claude/work/val_examples.parquet')
test_examples.to_parquet('/home/claude/work/test_examples.parquet')
print(f"[{time.time()-t0:.1f}s] Saved all example sets")
