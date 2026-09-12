"""Ilk blend denemesi (5-eski/3-yeni, LB=1.07361).
NOT: Bu script'in yerini scripts/05_final_blend_9old3new.py aldi (LB=1.07258,
daha iyi). Bu dosya sadece gecmis referans icin tutuluyor."""
import pandas as pd
import numpy as np
import lightgbm as lgb

FEATURE_COLS = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
]
OLD_MODELS = ['model_v4_fast2.txt', 'model_v4_seed101.txt', 'model_v4_seed202.txt',
              'model_v4_seed303.txt', 'model_v4_seed404.txt']
NEW_MODELS = ['model_optuna_seed42.txt', 'model_optuna_seed101.txt', 'model_optuna_seed303.txt']
OLD_WEIGHT, NEW_WEIGHT = 0.6, 0.4


def predict_ensemble(model_files, X):
    preds = [np.clip(np.expm1(lgb.Booster(model_file=f).predict(X)), 0, None) for f in model_files]
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    test_examples_full = pd.read_parquet('test_examples.parquet')
    X_test = test_examples_full[FEATURE_COLS]
    old_ens = predict_ensemble(OLD_MODELS, X_test)
    new_ens = predict_ensemble(NEW_MODELS, X_test)
    blend = OLD_WEIGHT * old_ens + NEW_WEIGHT * new_ens
    id_col = test_examples_full['id_y'] if 'id_y' in test_examples_full.columns else test_examples_full['id']
    submission = pd.DataFrame({'id': id_col.values, 'tuketim': blend})
    submission.to_csv('submission_v7_blend_legacy.csv', index=False)
    print("Legacy blend submission saved:", submission.shape)
