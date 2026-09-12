"""Devam ettirilebilir (resumable) Optuna hiperparametre arama scripti.
Kullanim: python3 03_optuna_search.py <yeni_deneme_sayisi>"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_log_error
import pickle, os, sys

FEATURE_COLS = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
]
CAT_COLS = ['il', 'bolge', 'ilce']


def load_slim(path, frac=None):
    df = pd.read_parquet(path)
    keep = [c for c in FEATURE_COLS if c in df.columns]
    extra = [c for c in ['target'] if c in df.columns]
    df = df[keep + extra].copy()
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    if frac:
        df = df.sample(frac=frac, random_state=0).reset_index(drop=True)
    return df


tr = load_slim('train_examples.parquet', frac=0.35)
va = load_slim('val_examples.parquet')
for c in CAT_COLS:
    tr[c] = tr[c].astype('category')
    va[c] = pd.Categorical(va[c], categories=tr[c].cat.categories)

X_tr, y_tr = tr[FEATURE_COLS], tr['target']
X_va, y_va = va[FEATURE_COLS], va['target']
true_va = np.expm1(y_va)


def objective(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'num_threads': 1, 'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.18),
        'num_leaves': trial.suggest_int('num_leaves', 24, 80),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 600),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': 1,
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 5.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 63, 150),
    }
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS, params={'max_bin': params['max_bin']})
    dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=350, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(35), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(X_va, num_iteration=m.best_iteration)), 0, None)
    rmsle = np.sqrt(mean_squared_log_error(true_va, pred))
    print(f"trial {trial.number}: rmsle={rmsle:.5f} params={params}")
    return rmsle


STUDY_PATH = 'optuna_study.pkl'
if os.path.exists(STUDY_PATH):
    with open(STUDY_PATH, 'rb') as f:
        study = pickle.load(f)
else:
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4
study.optimize(objective, n_trials=n_trials)
with open(STUDY_PATH, 'wb') as f:
    pickle.dump(study, f)
print(f"Total trials: {len(study.trials)}  Best: {study.best_value}  {study.best_params}")
