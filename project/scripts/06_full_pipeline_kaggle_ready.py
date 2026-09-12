"""
================================================================================
Trafo Bazli Elektrik Tuketimi Tahmini - Kaggle Ortaminda Calistirma Scripti
================================================================================

Bu script, yerel (kaynak kisitli) ortamda gelistirilen ve dogrulanan en iyi
pipeline'i icerir. Kaggle'in daha guclu hesaplama kaynaklarindan faydalanarak:
  1. Tam feature engineering (ufuk-farkinda tasarim + tum dogrulanmis feature'lar)
  2. TAM YAKINSAMAYA KADAR egitim (erken durdurma ile, round siniri yok)
  3. Coklu-seed ensemble
uretir.

KULLANIM (Kaggle Notebook'ta):
1. Yarisma verisini (train.csv, test.csv, sample_submission.csv) Kaggle
   notebook'unuza "Add Data" ile ekleyin (veya yarismanin kendi veri setini kullanin).
2. Asagidaki DATA_DIR degiskenini Kaggle'daki veri yolunuza gore guncelleyin
   (genelde '/kaggle/input/<dataset-adi>/').
3. Bu dosyanin tamamini bir Kaggle notebook hucresine yapistirip calistirin.
4. Cikti: /kaggle/working/submission.csv

ONEMLI - NEDEN BU SCRIPT?
Yerel gelistirme ortamimizda (1 CPU, sinirli sure) modelleri genelde 800-1050
boosting round'unda, erken durdurmaya ulasmadan kesmek zorunda kaldik (loglar
"Did not meet early stopping" diyordu - yani model hala iyilesiyordu). Bu script
Kaggle'in daha guclu/uzun-sureli ortaminda modelin GERCEKTEN yakinsamasina izin
verir (early_stopping_rounds=150, num_boost_round=5000 - pratikte cok daha erken
duracaktir ama tavan cok daha yuksek).

DOGRULANMIS SONUC (yerel, kisitli round ile): LB = 1.05562
Bu scriptin GERCEKTEN yakinsamis modellerle daha da iyi sonuc vermesi bekleniyor.
================================================================================
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error
import time
import os

t0 = time.time()

# ==============================================================================
# 0. AYARLAR - Kaggle'da veri yolunu buraya gore guncelleyin
# ==============================================================================
DATA_DIR = '/kaggle/input/YOUR-DATASET-NAME'   # <-- BURAYI GUNCELLEYIN
OUTPUT_DIR = '/kaggle/working'

# Eger yerel/farkli bir ortamda test ediyorsaniz:
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'   # ayni klasorde train.csv/test.csv varsayilir
    OUTPUT_DIR = '.'

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]   # istediginiz kadar seed ekleyebilirsiniz
NUM_BOOST_ROUND = 5000       # tavan - pratikte early stopping cok daha once durur
EARLY_STOPPING_ROUNDS = 150  # onceki 70'ten daha sabirli - tam yakinsama icin
LEARNING_RATE = 0.06         # onceki 0.12'den daha dusuk - daha kaliteli/yavas yakinsama

print(f"[{time.time()-t0:.1f}s] Baslatildi. DATA_DIR={DATA_DIR}")

# ==============================================================================
# 1. VERI YUKLEME VE TEMEL HAZIRLIK
# ==============================================================================
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
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
train['log_ratio'] = np.log1p((train['tuketim'].clip(lower=0) / train['guc']).clip(lower=0))
train['prefix6'] = train['tanim'].where(train['tanim'].str.match(r'^\d+$'), None).str[:6]
train = train.sort_values(['tanim', 'tarih']).reset_index(drop=True)

print(f"[{time.time()-t0:.1f}s] Yuklendi. train={train.shape} test={test.shape}")

# ==============================================================================
# 2. SNAPSHOT (ANLIK GORUNTU) FEATURE'LARI - her gercek kayit satiri icin
#    o tarihe kadarki (dahil) tum gecmisi ozetler
# ==============================================================================
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

train['dow_tmp'] = train['tarih'].dt.dayofweek
train['snap_dow_mean'] = (train.groupby(['tanim', 'dow_tmp'])['log_tuketim']
                           .expanding().mean().reset_index(level=[0, 1], drop=True)).values

train['cal_month_tmp'] = train['tarih'].dt.month
train['snap_month_mean'] = (train.groupby(['tanim', 'cal_month_tmp'])['log_tuketim']
                             .expanding().mean().reset_index(level=[0, 1], drop=True)).values

first_seen = train.groupby('tanim')['tarih'].transform('min')
train['snap_age_days'] = (train['tarih'] - first_seen).dt.days
train['snap_date'] = train['tarih']

print(f"[{time.time()-t0:.1f}s] Snapshot feature'lari olusturuldu")

snapshot_cols = ['snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
                  'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days']
snapshot_table = train[['tanim', 'snap_date'] + snapshot_cols].sort_values(['tanim', 'snap_date']).reset_index(drop=True)

guc_edges = pd.qcut(train['guc'], 20, duplicates='drop').cat.categories
train['guc_bucket'] = pd.cut(train['guc'], bins=guc_edges)
tanim_meta = train[['tanim', 'guc', 'guc_bucket', 'il', 'bolge', 'ilce', 'prefix6']].drop_duplicates('tanim')

FEATURE_COLS = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
    'fb_knn7_guc_ilce', 'fb_ratio_scaled', 'fb_prefix_mean',
]
CAT_COLS = ['il', 'bolge', 'ilce']


# ==============================================================================
# 3. KESIM-TARIHI BAZLI ORNEK URETME FONKSIYONU
# ==============================================================================
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

    hist_im2 = hist.merge(tanim_meta[['tanim', 'ilce']], on='tanim', how='left', suffixes=('', '_m2'))
    ilce_ratio_mean = hist_im2.groupby('ilce')['log_ratio'].mean().reset_index()
    ilce_ratio_mean.columns = ['ilce', 'ilce_log_ratio_mean']

    hist_gb = hist.merge(tanim_meta[['tanim', 'guc_bucket', 'ilce']], on='tanim', how='left', suffixes=('', '_m'))
    guc_ilce_mean = hist_gb.groupby(['guc_bucket', 'ilce'], observed=True)['log_tuketim'].mean().reset_index()
    guc_ilce_mean.columns = ['guc_bucket', 'ilce', 'fb_guc_ilce_mean']

    # k-NN benzeri: ayni ilcede en yakin K=7 guc'e sahip bilinen trafolarin ortalamasi
    tanim_own_mean = hist.groupby('tanim')['log_tuketim'].mean().reset_index()
    tanim_own_mean.columns = ['tanim', 'own_mean']
    knn_ref = tanim_meta[['tanim', 'guc', 'ilce']].merge(tanim_own_mean, on='tanim', how='inner')
    knn_ref = knn_ref.sort_values(['ilce', 'guc'])
    knn_ref['knn7_mean'] = knn_ref.groupby('ilce')['own_mean'].transform(
        lambda s: s.rolling(7, center=True, min_periods=1).mean())
    knn_ref = knn_ref.sort_values('guc')

    # PREFIX (ID'nin ilk 6 hanesi) tabanli fallback - ayni proje/binada kurulmus
    # trafolar cok daha benzer tuketim gosteriyor (test'teki yeni trafolarin ~%94'u
    # train'de bir prefix eslesmesi buluyor)
    hist_px = hist.merge(tanim_meta[['tanim', 'prefix6']], on='tanim', how='left', suffixes=('', '_px'))
    prefix_mean = hist_px.dropna(subset=['prefix6']).groupby('prefix6')['log_tuketim'].agg(['mean', 'count']).reset_index()
    prefix_mean.columns = ['prefix6', 'fb_prefix_mean', 'fb_prefix_count']

    global_mean = hist['log_tuketim'].mean()
    global_log_ratio_mean = hist['log_ratio'].mean()

    merged = merged.merge(tanim_meta, on='tanim', how='left')
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

    own_prefix = merged['tanim'].where(merged['tanim'].str.match(r'^\d+$'), None).str[:6]
    merged['prefix6'] = merged['prefix6'].fillna(own_prefix) if 'prefix6' in merged.columns else own_prefix

    merged['cal_month'] = merged['tarih'].dt.month
    merged = merged.merge(ilce_month_mean, on=['ilce', 'cal_month'], how='left')
    merged = merged.merge(ilce_mean, on='ilce', how='left')
    merged = merged.merge(ilce_ratio_mean, on='ilce', how='left')
    merged = merged.merge(guc_ilce_mean, on=['guc_bucket', 'ilce'], how='left')
    merged = merged.merge(prefix_mean, on='prefix6', how='left')

    if len(knn_ref) > 0:
        merged['guc'] = merged['guc'].astype('float64')
        knn_ref2 = knn_ref.copy()
        knn_ref2['guc'] = knn_ref2['guc'].astype('float64')
        merged['ilce'] = merged['ilce'].astype(str)
        knn_ref2['ilce'] = knn_ref2['ilce'].astype(str)
        merged = merged.sort_values('guc')
        merged = pd.merge_asof(
            merged, knn_ref2[['guc', 'ilce', 'knn7_mean']].rename(columns={'knn7_mean': 'fb_knn_guc_ilce'}),
            on='guc', by='ilce', direction='nearest'
        )
    else:
        merged['fb_knn_guc_ilce'] = np.nan
    merged['fb_guc_ilce_mean'] = merged['fb_knn_guc_ilce'].fillna(merged['fb_guc_ilce_mean'])

    merged['ilce_log_ratio_mean'] = merged['ilce_log_ratio_mean'].fillna(global_log_ratio_mean)
    implied_tuketim = np.expm1(merged['ilce_log_ratio_mean']).clip(lower=0) * merged['guc']
    merged['fb_ratio_scaled'] = np.log1p(implied_tuketim.clip(lower=0))

    merged['fb_ilce_month_mean'] = merged['fb_ilce_month_mean'].fillna(merged['fb_ilce_mean']).fillna(global_mean)
    merged['fb_ilce_mean'] = merged['fb_ilce_mean'].fillna(global_mean)
    merged['fb_guc_ilce_mean'] = merged['fb_guc_ilce_mean'].fillna(merged['fb_ilce_mean']).fillna(global_mean)
    merged['fb_knn7_guc_ilce'] = merged['fb_guc_ilce_mean']  # ayni sutun kullanilir

    merged['fb_prefix_count'] = merged['fb_prefix_count'].fillna(0)
    merged.loc[merged['fb_prefix_count'] < 3, 'fb_prefix_mean'] = np.nan
    merged['fb_prefix_mean'] = merged['fb_prefix_mean'].fillna(merged['fb_guc_ilce_mean']).fillna(merged['fb_ilce_mean']).fillna(global_mean)

    merged['snap_exp_std'] = merged['snap_exp_std'].fillna(0)
    merged['snap_exp_min'] = merged['snap_exp_min'].fillna(merged['snap_exp_mean'])
    merged['snap_exp_max'] = merged['snap_exp_max'].fillna(merged['snap_exp_mean'])
    merged['snap_exp_count'] = merged['snap_exp_count'].fillna(0)
    merged['is_new_tanim'] = (merged['snap_exp_count'] == 0).astype(int)

    for c in ['snap_exp_mean', 'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean']:
        merged[c] = merged[c].fillna(merged['fb_guc_ilce_mean']).fillna(merged['fb_ilce_month_mean']) \
                              .fillna(merged['fb_ilce_mean']).fillna(global_mean)

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


# ==============================================================================
# 4. COGALTILMIS EGITIM SETI (10 kesim noktasi) + VALIDASYON + TEST
# ==============================================================================
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
    print(f"[{time.time()-t0:.1f}s] kesim {c.date()}: {len(ex)} ornek")

train_examples = pd.concat(aug_frames, ignore_index=True)
print(f"[{time.time()-t0:.1f}s] Toplam egitim ornegi: {len(train_examples)}")

val_cutoff = pd.Timestamp('2025-03-31')
val_horizon_end = val_cutoff + pd.Timedelta(days=MAX_HORIZON)
val_targets = train[(train['tarih'] > val_cutoff) & (train['tarih'] <= val_horizon_end)][['tanim', 'tarih', 'log_tuketim']]
val_examples = build_examples(val_cutoff, val_targets[['tanim', 'tarih']])
val_examples['target'] = val_targets.set_index(['tanim', 'tarih']).loc[
    list(zip(val_examples['tanim'], val_examples['tarih']))]['log_tuketim'].values
print(f"[{time.time()-t0:.1f}s] Validasyon ornegi: {len(val_examples)}")

test_cutoff = train['tarih'].max()
test_own = test[['tanim', 'tarih', 'guc', 'il', 'bolge', 'ilce']].rename(
    columns={'guc': 'own_guc', 'il': 'own_il', 'bolge': 'own_bolge', 'ilce': 'own_ilce'})
test_examples = build_examples(test_cutoff, test_own)
test_examples = test_examples.merge(test[['tanim', 'tarih', 'id']], on=['tanim', 'tarih'], how='left')
print(f"[{time.time()-t0:.1f}s] Test ornegi: {len(test_examples)} (beklenen: {len(test)})")

for c in CAT_COLS:
    all_cats = pd.concat([train_examples[c], val_examples[c], test_examples[c]]).astype('category').cat.categories
    train_examples[c] = pd.Categorical(train_examples[c], categories=all_cats)
    val_examples[c] = pd.Categorical(val_examples[c], categories=all_cats)
    test_examples[c] = pd.Categorical(test_examples[c], categories=all_cats)

# ==============================================================================
# 5. MODEL EGITIMI - TAM YAKINSAMAYA KADAR (erken durdurma ile), COKLU SEED
# ==============================================================================
# bellek verimliligi icin float32'ye indirgeme (Kaggle'da bile RAM tasarrufu saglar)
for _c in FEATURE_COLS:
    if train_examples[_c].dtype == 'float64':
        train_examples[_c] = train_examples[_c].astype('float32')
    if val_examples[_c].dtype == 'float64':
        val_examples[_c] = val_examples[_c].astype('float32')
    if test_examples[_c].dtype == 'float64':
        test_examples[_c] = test_examples[_c].astype('float32')

X_tr, y_tr = train_examples[FEATURE_COLS], train_examples['target']
X_va, y_va = val_examples[FEATURE_COLS], val_examples['target']
X_te = test_examples[FEATURE_COLS]

params_base = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': LEARNING_RATE,
    'num_leaves': 48,
    'min_data_in_leaf': 300,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'max_bin': 90,
    'lambda_l2': 1.0,
    'verbose': -1,
}

dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)

models = []
val_preds = []
for seed in SEEDS:
    params = dict(params_base)
    params['seed'] = seed
    t1 = time.time()
    m = lgb.train(
        params, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dva], valid_names=['val'],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(200)]
    )
    pred = np.clip(np.expm1(m.predict(X_va, num_iteration=m.best_iteration)), 0, None)
    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_va), pred))
    print(f"[{time.time()-t0:.1f}s] seed={seed}  best_iter={m.best_iteration}  "
          f"val RMSLE={rmsle:.5f}  (sure: {time.time()-t1:.0f}s)")
    models.append(m)
    val_preds.append(pred)
    m.save_model(f'{OUTPUT_DIR}/model_seed{seed}.txt')

ens_val_pred = np.mean(val_preds, axis=0)
ens_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_va), ens_val_pred))
print(f"\n>>> {len(SEEDS)}-seed ENSEMBLE Validasyon RMSLE: {ens_rmsle:.5f}")

# ==============================================================================
# 6. TEST TAHMINLERI VE SUBMISSION
# ==============================================================================
test_preds = [np.clip(np.expm1(m.predict(X_te, num_iteration=m.best_iteration)), 0, None) for m in models]
test_pred = np.mean(test_preds, axis=0)

submission = pd.DataFrame({'id': test_examples['id'].values, 'tuketim': test_pred})
submission.to_csv(f'{OUTPUT_DIR}/submission.csv', index=False)
print(f"\n[{time.time()-t0:.1f}s] Submission kaydedildi: {submission.shape}")
print(submission['tuketim'].describe())
