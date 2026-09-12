"""
FINAL uretim yaklasimi (KANITLANMIS EN IYI SONUC, LB=1.07258): 9-seed ESKI
ensemble + 3-seed YENI (Optuna) ensemble, 0.6/0.4 agirlikla harmanlanir.

GECMIS:
- 5-seed ESKI (elle secilmis params):           LB=1.0748
- 3-seed YENI (Optuna params):                  LB=1.0800 (val'de iyi, LB'de kotu -> overfit)
- blend(5-eski/3-yeni, 0.6/0.4):                LB=1.07361
- blend(9-eski/3-yeni, 0.6/0.4):                LB=1.07258  <-- BU SCRIPT, EN IYI

Bu script, scripts/02_train_ensemble_production.py'nin 9 seed (42,101,202,303,
404,505,555,606,666) uretecek sekilde genisletilmis halini ve
scripts/04_final_blend_production.py'nin blend mantigini birlestirir.
"""
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

OLD_PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.12, 'num_leaves': 48, 'min_data_in_leaf': 300,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'max_bin': 90, 'lambda_l2': 1.0, 'verbose': -1, 'num_threads': 1,
}
OLD_SEEDS = [42, 101, 202, 303, 404, 505, 555, 606, 666]   # 5 -> 9 seed genisletildi

NEW_PARAMS = {
    'objective': 'regression', 'metric': 'rmse',
    'learning_rate': 0.10494481426168349, 'num_leaves': 78, 'min_data_in_leaf': 466,
    'feature_fraction': 0.8095304694689628, 'bagging_fraction': 0.6546065241548528,
    'bagging_freq': 1, 'lambda_l2': 0.18408992080552522, 'max_bin': 68,
    'verbose': -1, 'num_threads': 1,
}
NEW_SEEDS = [42, 101, 303]  # seed=202 elendi (erken durup zayif cikmisti)

NUM_ROUNDS = 1050
EARLY_STOP = 70
OLD_WEIGHT, NEW_WEIGHT = 0.6, 0.4


def load_slim(path):
    df = pd.read_parquet(path)
    keep = [c for c in FEATURE_COLS if c in df.columns]
    extra = [c for c in ['target', 'id', 'id_y'] if c in df.columns]
    df = df[keep + extra].copy()
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')
    return df


def train_pool(params_base, seeds, train_examples, val_examples, prefix):
    models = []
    for seed in seeds:
        params = dict(params_base); params['seed'] = seed
        model_path = f'model_{prefix}_seed{seed}.txt'
        import os
        if os.path.exists(model_path):
            models.append(lgb.Booster(model_file=model_path))
            continue
        X_tr, y_tr = train_examples[FEATURE_COLS], train_examples['target']
        X_va, y_va = val_examples[FEATURE_COLS], val_examples['target']
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS,
                           params={'max_bin': params_base['max_bin']})
        dva = lgb.Dataset(X_va, y_va, categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=NUM_ROUNDS,
                       valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(300)])
        m.save_model(model_path)
        models.append(m)
    return models


def predict_ensemble(models, X):
    preds = [np.clip(np.expm1(m.predict(X)), 0, None) for m in models]
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    train_examples = load_slim('train_examples.parquet')
    val_examples = load_slim('val_examples.parquet')
    test_examples_full = pd.read_parquet('test_examples.parquet')

    old_models = train_pool(OLD_PARAMS, OLD_SEEDS, train_examples, val_examples, 'v4')
    new_models = train_pool(NEW_PARAMS, NEW_SEEDS, train_examples, val_examples, 'optuna')

    val_true = np.expm1(val_examples['target'])
    old_ens_val = predict_ensemble(old_models, val_examples[FEATURE_COLS])
    new_ens_val = predict_ensemble(new_models, val_examples[FEATURE_COLS])
    blend_val = OLD_WEIGHT * old_ens_val + NEW_WEIGHT * new_ens_val
    print(f"Validation RMSLE (final blend): {np.sqrt(mean_squared_log_error(val_true, blend_val)):.5f}")

    X_test = test_examples_full[FEATURE_COLS]
    old_ens_test = predict_ensemble(old_models, X_test)
    new_ens_test = predict_ensemble(new_models, X_test)
    blend_test = OLD_WEIGHT * old_ens_test + NEW_WEIGHT * new_ens_test

    id_col = test_examples_full['id_y'] if 'id_y' in test_examples_full.columns else test_examples_full['id']
    submission = pd.DataFrame({'id': id_col.values, 'tuketim': blend_test})
    submission.to_csv('submission_production_final.csv', index=False)
    print("Submission saved (LB=1.07258):", submission.shape)
