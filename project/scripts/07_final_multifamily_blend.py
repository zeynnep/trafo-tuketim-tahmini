"""
NIHAI uretim yaklasimi (LB=1.054, bu yarismadaki en iyi kanitlanmis sonuc):
5 farkli model ailesinin agirlikli ensemble'i.

Tek bir "en iyi" konfigurasyona guvenmek yerine (ki bu Optuna denemesinde
val'de iyi/LB'de kotu cikarak bizi yanilmisti), birbirinden farkli ozellik
setleri ve hiperparametrelerle egitilmis COKLU model ailesini harmanlamak
tutarli sekilde daha iyi ve daha guvenilir sonuc verdi.

Model Aileleri:
1. "Eski" (elle secilmis params, temel ozellikler, k=1 fallback) - 10 seed
2. "Optuna" (sistematik hiperparametre aramasi) - 3 seed
3. "Additive-k7" (+ k=7 komsu ortalamasi fallback) - 5 seed
4. "Weighted-sample" (+ test kompozisyonuna gore agirlikli egitim) - 3 seed
5. "Prefix ailesi" (+ oran + ID-prefix + prefix-oran birlesimi) - cesitlendirilmis

NOT: Prefix ailesini cesitlendirirken DIKKATLI olun - "seed101" tek basina
zayif cikmisti (ayri deneyde kanitlandi), fazla agirlik/kotu kombinasyon
public leaderboard'da ciddi kotulesmeye yol acabilir (bkz. notebook 06,
bolum 6.5 - "Onemli Ders: Public/Private Leaderboard Varyansi"). Asagidaki
agirliklar, LB'de kanitlanmis en iyi (1.054) konfigurasyonu yansitir.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

FEATURE_COLS_BASE = [
    'guc', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'is_weekend', 'quarter',
    'il', 'bolge', 'ilce',
    'horizon_days', 'recency_days',
    'snap_exp_count', 'snap_exp_mean', 'snap_exp_std', 'snap_exp_min', 'snap_exp_max',
    'snap_last_val', 'snap_roll30', 'snap_roll90', 'snap_dow_mean', 'snap_month_mean', 'snap_age_days',
    'is_new_tanim', 'fb_ilce_month_mean', 'fb_ilce_mean', 'fb_guc_ilce_mean',
]
FEATURE_COLS_ADDK7 = FEATURE_COLS_BASE + ['fb_knn7_guc_ilce']
FEATURE_COLS_RATIO = FEATURE_COLS_ADDK7 + ['fb_ratio_scaled']
FEATURE_COLS_PREFIX = FEATURE_COLS_RATIO + ['fb_prefix_mean']
FEATURE_COLS_PREFIXRATIO = FEATURE_COLS_PREFIX + ['fb_prefix_ratio_scaled']

OLD_MODELS = ['model_v4_fast2.txt', 'model_v4_seed101.txt', 'model_v4_seed202.txt',
              'model_v4_seed303.txt', 'model_v4_seed404.txt', 'model_v4_seed505.txt',
              'model_v4_seed555.txt', 'model_v4_seed606.txt', 'model_v4_seed666.txt',
              'model_v4_seed808.txt']
NEW_MODELS = ['model_optuna_seed42.txt', 'model_optuna_seed101.txt', 'model_optuna_seed303.txt']
ADDK7_MODELS = ['model_addk7_seed42.txt', 'model_addk7_seed101.txt', 'model_addk7_seed303.txt',
                'model_addk7_seed404.txt', 'model_addk7_seed505.txt']
WEIGHTED_MODELS = ['model_weighted_seed42.txt', 'model_weighted_seed101.txt', 'model_weighted_seed202.txt']
# Prefix ailesi: yalnizca DOGRULANMIS iyi bilesenler (seed101 haric!)
PREFIX_MODELS = [('model_prefix_seed42.txt', FEATURE_COLS_PREFIX),
                  ('model_prefixratio_seed42.txt', FEATURE_COLS_PREFIXRATIO)]


def predict_ensemble(model_files, X):
    preds = [np.clip(np.expm1(lgb.Booster(model_file=f).predict(X)), 0, None) for f in model_files]
    return np.mean(preds, axis=0)


if __name__ == '__main__':
    test_examples_full = pd.read_parquet('test_examples.parquet')  # prefix_ratio_feat pipeline'indan

    old_ens = predict_ensemble(OLD_MODELS, test_examples_full[FEATURE_COLS_BASE])
    new_ens = predict_ensemble(NEW_MODELS, test_examples_full[FEATURE_COLS_BASE])
    addk7_ens = predict_ensemble(ADDK7_MODELS, test_examples_full[FEATURE_COLS_ADDK7])
    weighted_ens = predict_ensemble(WEIGHTED_MODELS, test_examples_full[FEATURE_COLS_ADDK7])
    prefix_preds = [np.clip(np.expm1(lgb.Booster(model_file=f).predict(test_examples_full[cols])), 0, None)
                     for f, cols in PREFIX_MODELS]
    prefix_ens = np.mean(prefix_preds, axis=0)

    proven = 0.6 * old_ens + 0.4 * new_ens
    base = 0.85 * (0.4 * proven + 0.6 * addk7_ens) + 0.15 * weighted_ens
    final = 0.7 * base + 0.3 * prefix_ens  # LB=1.054'u ureten agirliklar

    id_col = test_examples_full['id_y'] if 'id_y' in test_examples_full.columns else test_examples_full['id']
    submission = pd.DataFrame({'id': id_col.values, 'tuketim': final})
    submission.to_csv('submission_final_1054.csv', index=False)
    print("Nihai submission kaydedildi (LB=1.054):", submission.shape)
