"""Eski (elle secilmis) konfigurasyonla 5-9 seed LightGBM ensemble egitimi.
Detaylar icin scripts/05_final_blend_9old3new.py'ye bakin (nihai/birlesik script)."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error

FEATURE_COLS = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
]
CAT_COLS = ['il', 'bolge', 'ilce']
PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.12, 'num_leaves': 48, 'min_data_in_leaf': 300,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'max_bin': 90, 'lambda_l2': 1.0, 'verbose': -1, 'num_threads': 1,
}
SEEDS = [42, 101, 202, 303, 404, 505, 555, 606, 666]
NUM_ROUNDS = 1050
EARLY_STOP = 70


def load_slim(path):
    df = pd.read_parquet(path)
    keep = [c for c in FEATURE_COLS if c in df.columns]
    extra = [c for c in ['target', 'id', 'id_y'] if c in df.columns]
    df = df[keep + extra].copy()
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    return df


if __name__ == '__main__':
    train_examples = load_slim('train_examples.parquet')
    val_examples = load_slim('val_examples.parquet')
    test_examples_full = pd.read_parquet('test_examples.parquet')

    models, val_preds = [], []
    for seed in SEEDS:
        params = dict(PARAMS); params['seed'] = seed
        X_tr, y_tr = train_examples[FEATURE_COLS], train_examples['target']
        X_va, y_va = val_examples[FEATURE_COLS], val_examples['target']
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(300)])
        m.save_model(f'model_v4_seed{seed}.txt')
        pred = np.clip(np.expm1(m.predict(X_va, num_iteration=m.best_iteration)), 0, None)
        val_preds.append(pred)
        models.append(m)

    val_true = np.expm1(val_examples['target'])
    ens_pred = np.mean(val_preds, axis=0)
    print(f"{len(SEEDS)}-seed ensemble validation RMSLE: {np.sqrt(mean_squared_log_error(val_true, ens_pred)):.5f}")

    test_X = test_examples_full[FEATURE_COLS]
    test_preds = [np.clip(np.expm1(m.predict(test_X, num_iteration=m.best_iteration)), 0, None) for m in models]
    test_pred = np.mean(test_preds, axis=0)
    id_col = test_examples_full['id_y'] if 'id_y' in test_examples_full.columns else test_examples_full['id']
    submission = pd.DataFrame({'id': id_col.values, 'tuketim': test_pred})
    submission.to_csv('submission_old_9seed.csv', index=False)
    print("Submission saved:", submission.shape)
