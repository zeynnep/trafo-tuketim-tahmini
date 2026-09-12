# Trafo Bazlı Elektrik Tüketimi Tahmini

Gdz ve Adm Elektrik yarışması için trafo bazlı günlük elektrik tüketimi tahmin modeli.
**Metrik:** RMSLE (Root Mean Squared Logarithmic Error)

## 🏁 Nihai Sonuç

**492 takım içinden 147. sıra — Private Leaderboard skoru: 1.05084**

Public leaderboard'da en iyi denemelerimiz 1.054 civarındaydı; son (riskli) denemede
public skor 1.55'e sıçramıştı ama bunun **örneklem gürültüsü** olduğu şüphesi
doğrulandı — gerçek (private) skor 1.05084, en iyi public sonucumuza çok yakın,
hatta biraz daha iyi çıktı (bkz. Notebook 06, "Public/Private Leaderboard Varyansı").

## Klasör Yapısı

```
.
├── data/
│   ├── train.csv, test.csv, sample_submission.csv   # ham veri
│   └── processed/                                     # notebook 2 çalıştırılınca üretilir
├── notebooks/
│   ├── 01_eda.ipynb                              # Keşifsel veri analizi + görselleştirmeler
│   ├── 02_feature_engineering.ipynb              # Ufuk-farkında feature engineering + trend deneyi
│   ├── 03_modelleme.ipynb                        # LightGBM seed-ensemble eğitimi, validasyon
│   ├── 04_hiperparametre_optimizasyonu.ipynb     # Optuna ile sistematik hiperparametre taraması
│   ├── 05_coklu_pencere_dogrulama.ipynb          # 2. bağımsız validasyon penceresi + 9-seed genişletme
│   └── 06_ileri_feature_muhendisligi.ipynb       # Oran & ID-prefix keşfi, 3-pencere doğrulama, son dersler
├── scripts/
│   ├── 01_feature_engineering_production.py      # tam-ölçek üretim kodu (10 kesim, 3.4M satır)
│   ├── 02_train_ensemble_production.py           # eski (10-seed) config eğitimi
│   ├── 03_optuna_search.py                       # devam ettirilebilir Optuna arama scripti
│   ├── 04_final_blend_production.py              # ilk blend denemesi (referans)
│   ├── 05_final_blend_9old3new.py                # 9-eski/3-yeni blend (referans)
│   ├── 06_full_pipeline_kaggle_ready.py          # Kaggle'da tam-yakınsama için hazırlanmış tek-dosya script
│   └── 07_final_multifamily_blend.py             # NİHAİ üretim scripti (5 model ailesi blend, LB=1.054)
└── submission/
    └── submission_final_1054.csv                 # YARIŞMAYA GÖNDERİLEN, EN İYİ (public LB=1.054) submission
```

## Çalıştırma Sırası

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_hiperparametre_optimizasyonu.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modelleme.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_coklu_pencere_dogrulama.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_ileri_feature_muhendisligi.ipynb
```

Üretim submission'ı için: tüm model dosyaları ve parquet'ler aynı klasörde
olacak şekilde `scripts/` altındaki dosyalar sırayla çalıştırılmalı, son
adım `07_final_multifamily_blend.py`.

**Kaggle'da tam-yakınsama ile çalıştırmak isteyenler için:** `scripts/06_full_pipeline_kaggle_ready.py`
tek başına, uçtan uca çalışan bir dosyadır — sadece `DATA_DIR` değişkenini
güncelleyip Kaggle notebook'una yapıştırmak yeterlidir.

## Yöntem Özeti

### 1. EDA (`01_eda.ipynb`)
- Güçlü yaz mevsimselliği (Mayıs en düşük, Temmuz zirve — test dönemi Nisan-Temmuz'u kapsıyor)
- Test trafolarının ~%29'u train'de hiç yok → güç+lokasyon bazlı fallback şart
- Uç değerler ham ortalamayı bozuyor → log1p + medyan tabanlı istatistikler
- Düzensiz raporlama (gap'ler) → "en son bilinen değer + yaşı" yaklaşımı

### 2. Feature Engineering (`02_feature_engineering.ipynb`)
**Kritik tasarım kararı — "ufuk-farkında" (horizon-aware) feature'lar:** Test seti,
train'in bittiği tarihten (2026-03-31) sonraki 122 gün için tahmin istiyor ve bu
süreçte günlük gerçek değer geri beslemesi yok. Bu yüzden tüm feature'lar **kesim
tarihinde donduruluyor** ve modele `horizon_days` + `recency_days` besleniyor.

- 10 kesim noktası (aylık: Nisan 2025 → Ocak 2026), ~3.4M çoğaltılmış eğitim örneği
- Yeni trafo (test'in %29'u) fallback'i: aynı ilçedeki en yakın güce sahip bilinen
  trafo (k-NN benzeri, `merge_asof` ile)

### 3. Hiperparametre Optimizasyonu (`04_hiperparametre_optimizasyonu.ipynb`)
Optuna (TPE sampler) ile 17 denemelik sistematik tarama. Validasyonda daha iyi
bir konfigürasyon bulundu, ANCAK LB'de kötüleşti — **tek pencereye aşırı uyum**
riskinin somut bir örneği (bkz. bölüm "Önemli Dersler").

### 4. Modelleme (`03_modelleme.ipynb`)
- LightGBM, `log1p(tuketim)` hedefi üzerinde RMSE minimizasyonu (RMSLE'yi optimize eder)
- Validasyon: kesim=2025-03-31, hedef=2025-04-01..2025-07-31, sızıntısız
- Seed-ensemble: her seed'in validasyon performansı ayrı ayrı kontrol ediliyor

### 5. Çoklu Pencere Doğrulama (`05_coklu_pencere_dogrulama.ipynb`)
İkinci, bağımsız bir validasyon penceresi (kesim=2025-02-28) kuruldu. Bu,
Optuna konfigürasyonunun **val1'e özgü aşırı uyum** sağladığını doğruladı.

### 6. İleri Feature Mühendisliği (`06_ileri_feature_muhendisligi.ipynb`)
**En büyük bulgu:** Hatanın büyük kısmı yeni trafolardan geliyor (bilinen
trafolarda RMSLE~0.6, yeni trafolarda ~1.6-2.3). Bunu iyileştirmek için:
- **Oran (tüketim/güç) tabanlı fallback:** İlçe içi varyansı 3 kat azaltıyor,
  **3 bağımsız pencerede de tutarlı iyileşme** — oturumun en güvenilir bulgusu
- **ID-prefix (trafo ID'sinin ilk 6 hanesi) tabanlı fallback:** Aynı proje/binada
  kurulmuş trafoları gruplar; test'teki yeni trafoların %94'ü train'de bir
  prefix eşleşmesi buluyor. val1'de çok güçlü, val2/val3'te karışık — yine de
  gerçek LB'de küçük-adımlı, temkinli ağırlıklarla sürekli iyileşme sağladı.
- 3. bir üçüncü doğrulama penceresi (kesim=2025-01-31) eklendi

## Denenen Versiyonlar ve Leaderboard Geçmişi

| Versiyon | Yaklaşım | LB Skoru |
|---|---|---|
| v1 | Naif istatistik tabanlı baseline | 1.36 |
| v2 | "Dünkü değer" ağırlıklı lag/rolling | 1.55 |
| v3 | Ufuk-farkında (horizon-aware) tasarım | 1.47 → 1.26 |
| v4 | +10 kesim noktası, k-NN fallback, 3-seed ensemble | 1.07 |
| v5 | 5-seed ensemble (elle seçilmiş parametreler) | 1.0748 |
| v6 | Optuna (17 deneme) + 3-seed ensemble | 1.080 (LB'de kötüleşti ⚠️) |
| v7 | blend(5-eski/3-yeni, 0.6/0.4) | 1.07361 |
| v8 | +weighted-sample, 9-seed genişletme | 1.05839 |
| v9 | +oran+prefix fallback, küçük-adımlı ağırlıklandırma | 1.05562 |
| **v10** | **5 model ailesinin çeşitlendirilmiş blend'i** | **1.054 (public)** |
| v11-deneme | Prefix ağırlığı %40'a çıkarıldı + zayıf seed tekrar eklendi | 1.55 (public) → **1.05084 (private, nihai)** ✅ |

**Not:** v11-deneme'nin public skoru (1.55) yanıltıcıydı — private leaderboard'da
(gerçek, tam test seti üzerinde) 1.05084 çıktı, yani gerçekte v10'dan bile hafifçe
daha iyiydi. Bu, public/private varyans dersimizin en net kanıtı.

## Önemli Dersler
1. **Tek validasyon penceresi yanıltıcı olabilir** — Optuna'nın bulduğu "iyileşme"
   hem LB'de hem ikinci bağımsız pencerede kötü çıktı.
2. **Çeşitlilik (diversity) tek başına zayıf bir modeli bile faydalı kılabilir** —
   çok-aileli blend, tek bir "en iyi" konfigürasyondan daha güvenilir sonuç verdi.
3. **Seed sayısını artırmak** küçük ama tutarlı bir kazanç sağlıyor — ama körlemesine
   her seed'i eklemek yerine her birinin validasyon performansı kontrol edilmeli.
4. **Gerçek LB geri bildirimi, simülasyon pencerelerinden daha güvenilir** —
   ama **public leaderboard küçük bir örneklem üzerinden hesaplanıyorsa**, o da
   tek başına tam güvenilir olmayabilir; yüksek varyanslı alt gruplar (bizim
   durumumuzda yeni trafolar) küçük örneklemde skoru sert şekilde sarsabilir.
   **Bu yarışmada bu ders somut olarak doğrulandı:** son denemenin public skoru
   (1.55) endişe vericiydi ama private skoru (1.05084) aslında iyiydi.
5. Her yeni feature otomatik olarak iyileştirmez (trend feature örneği) —
   doğrulanmalı, gerekirse reddedilmeli.
6. **Kendi geçmiş bulgularınıza sadık kalın** — daha önce zayıf çıktığı kanıtlanmış
   bir bileşeni "belki bu sefer işe yarar" diyerek tekrar denemek risklidir.

## Sonraki Adımlar / Fikirler
- Gerçek k-fold / çoklu-kesim çapraz doğrulama ile hiperparametre seçimi
- CatBoost/farklı model aileleri ile daha fazla çeşitlilik
- Yeni trafolar için özel bir alt-model
- Hava durumu / resmi tatil verisi (yarışma açıklamasında bahsediliyor,
  sağlanan dosyalarda yok)

## Not
`data/processed/` klasörü (feature engineering çıktıları) repoya dahil edilmedi —
`02_feature_engineering.ipynb` çalıştırıldığında otomatik olarak yeniden üretilir.
