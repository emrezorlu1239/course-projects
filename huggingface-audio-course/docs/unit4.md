# Ünite 4 — GTZAN müzik türü sınıflandırma (AST ince ayarı)

**Sonuç:** doğrulama doğruluğu **0.8978** (parça/chunk düzeyi, 450 parça), **0.9133** (klip düzeyi, 150 klip).
Ödev eşiği 0.87. Model: `MIT/ast-finetuned-audioset-10-10-0.4593` → 10 sınıf.

Kod: [`unit4_genre_classifier/src/data.py`](../unit4_genre_classifier/src/data.py),
[`train.py`](../unit4_genre_classifier/src/train.py), [`publish.py`](../unit4_genre_classifier/src/publish.py).

## 1. Kod ne yapıyor?

| Adım | Nerede | Ne yapıyor |
|---|---|---|
| Veri yükleme | `data.load_gtzan` | `marsyas/gtzan` deposunun otomatik parquet dalını okur; sesi ham bayt olarak alır (`decode=False`). `datasets` 5.x script ve torchcodec istemediği için böyle. |
| Çözme + örnekleme | `data.decode` | soundfile ile çöz → mono → 22.05 kHz'ten 16 kHz'e (librosa). AST 16 kHz ile ön-eğitildi. |
| Parçalama | `data.chunk` | 30 sn klip → 3 adet 10 sn parça. AST tam olarak 1024 kare ≈ 10.24 sn bekler. |
| Klip bazında bölme | `data.split_clips` | Bölme **klip indeksleri** üzerinde, türe göre tabakalı (%85/%15). Bir klibin 3 parçası ya hepsi eğitimde ya hepsi doğrulamada. `train.py` bunu `assert` ile kontrol eder. |
| Özellik önbelleği | `data.build_cache` | `ASTFeatureExtractor`: 128 mel kutulu log-mel fbank (25 ms pencere, 10 ms adım), sonra `(x - mean) / (2·std)` normalizasyonu. Sonuç float16 olarak `.npz` içine yazılır; her epoch'ta yeniden hesaplanmaz. |
| Artırma | `ChunkDataset` | SpecAugment: 2 frekans maskesi (≤24 kutu) + 2 zaman maskesi (≤96 kare), maskelenen değer 0 = veri setinin ortalaması. |
| Model | `train.main` | AudioSet başlığı (527 sınıf) atılır, yeni 10 sınıflık doğrusal başlık rastgele başlatılır (`ignore_mismatched_sizes=True`); tüm ağ ince ayarlanır. |
| Eğitim | `Trainer` | lr 3e-5, cosine çizelge, %10 warmup, 8 epoch, batch 8 × gradient accumulation 2 = 16, fp16, weight decay 0.01. |
| En iyi checkpoint | `train.main` | transformers 5.18'de `load_best_model_at_end` AST ağırlıklarını sessizce yükleyemiyor (anahtar adları dönüştürülmüyor). Bu yüzden en iyi checkpoint `from_pretrained` ile elle yüklenir ve son bir `evaluate()` çalıştırılır; model kartı bu son değerlendirmeyi yazar. |
| Rapor | `train.main` | Karmaşıklık matrisi PNG, sınıf bazında F1, klip düzeyi doğruluk (`results/unit4_*.json`). |

## 2. Matematik (sade dille)

### 2.1 Log-mel spektrogram
Ses, kısa pencerelere (25 ms) bölünür, her pencereye Fourier dönüşümü uygulanır → hangi frekansta ne kadar enerji var.
Frekans ekseni insan kulağına benzeyen **mel** ölçeğine 128 kutuyla sıkıştırılır, enerjinin logaritması alınır
(kulak da yüksekliği logaritmik algılar). 10 sn → 1024 zaman karesi × 128 mel kutusu = bir "görüntü".

### 2.2 Yamalama (patching) — kaç token var?
AST, bu 1024×128 görüntüyü ViT gibi 16×16 yamalara böler ama yamalar **örtüşür** (adım 10):

- frekans yönünde: (128 − 16) / 10 + 1 = **12**
- zaman yönünde: (1024 − 16) / 10 + 1 = **101**
- toplam 12 × 101 = **1212 yama** + `[CLS]` + `[DIST]` = **1214 token**, her biri 768 boyutlu vektör.

Self-attention maliyeti token sayısının karesiyle büyür (1214² ≈ 1.5 milyon çift) — 8 GB'a sığdırmak için fp16 ve batch 8 kullandık.

### 2.3 Çapraz entropi (cross-entropy)
Model her parça için 10 sayı (logit) z₁..z₁₀ üretir. Softmax bunları olasılığa çevirir:
p_k = e^{z_k} / Σ_j e^{z_j}. Doğru sınıf y ise kayıp **L = −log p_y**.

- Model doğru sınıfa 0.9 verirse L = 0.105; 0.1 verirse L = 2.30. Yanlış ve emin olmak çok pahalıdır.
- Label smoothing (2. koşuda 0.1) hedefi "%100 doğru sınıf" yerine "%91 doğru sınıf + kalan %9 öbür 9 sınıfa eşit" yapar; aşırı özgüveni azaltır. Burada yardımcı olmadı (0.889 < 0.898).

Gözlem: 4. epoch'tan sonra eğitim kaybı ~0'a inerken doğrulama kaybı 0.41'den 0.57'ye çıktı ama doğruluk 0.88'den 0.90'a yükseldi.
Yani model yanlış bildiği az sayıdaki örnekte **çok emin** yanlış yapıyor (kayıp büyür), ama doğru sayısı artıyor.

### 2.4 Doğruluk ne kadar güvenilir? (standart hata)
Doğruluk p, n bağımsız örnekte ölçülürse standart hata SE = √(p(1−p)/n).

- Klip düzeyi: p = 0.913, n = 150 → SE ≈ **0.023** → yaklaşık %95 aralık 0.87–0.96.
- Parça düzeyi: n = 450 → SE ≈ 0.014, ama aynı klibin 3 parçası bağımsız değil, gerçek belirsizlik bundan büyüktür.

Bu yüzden 0.87 eşiğinin hemen üstü riskli olurdu; ~0.90 hedefi bu yüzden konmuştu.

### 2.5 Neden klip bazında bölme?
Aynı 30 sn şarkının parçaları hem eğitimde hem doğrulamada olsaydı, model "bu şarkıyı" ezberleyerek doğrulamada
yapay yüksek skor alırdı (**sızıntı / leakage**). Klip bazında bölme bunu engeller. (GTZAN'da aynı sanatçının
farklı şarkıları yine de iki tarafta olabilir — bilinen bir veri seti sorunu, model kartında belirttik.)

## 3. Önceki projelerinle bağlantı

**AST hayvan sesi projesi** (`ast-animal-sound-classifier`, ESC-50 hayvan alt kümesi, test doğruluğu %86.25):

| | Hayvan sesleri (önceki) | GTZAN (bu ünite) |
|---|---|---|
| Ön-eğitimli ağırlık | aynı: `MIT/ast-finetuned-audioset-10-10-0.4593` | aynı |
| Girdi | 5 sn → 1024×128 (dolgulu) | 10 sn → 1024×128 (tam dolu) |
| Bölme | resmi ESC-50 fold'ları (sızıntısız) | klip bazında tabakalı bölme (sızıntısız) |
| Artırma | SpecAugment (24 frekans / 64 zaman) | SpecAugment (24 / 96) |
| Eğitim döngüsü | kendi PyTorch döngün + AMP | Hugging Face `Trainer` + fp16 |
| Sıfırdan vs transfer | transfer %86.25, sıfırdan %32.5 | yalnızca transfer |

Oradaki ana ders burada da geçerli: AudioSet ön-eğitimi olmadan 800 klipten bir Transformer öğrenmek neredeyse
imkânsız; transfer öğrenme ile 8 epoch'ta %90'a yaklaşıyoruz.

**cGAN projesi** ile ortak nokta: orada Discriminator da bir **ikili sınıflandırıcıdır** ve ikili çapraz entropi ile
eğitilir (gerçek/sahte). Burada aynı fikrin 10 sınıflı hâlini kullanıyoruz. One-sided label smoothing (cGAN) ile
buradaki label smoothing aynı amaca hizmet eder: sınıflandırıcının aşırı emin olmasını engellemek.
