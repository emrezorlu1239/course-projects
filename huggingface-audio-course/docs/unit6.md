# Ünite 6 — SpeechT5 metinden konuşmaya (Felemenkçe, VoxPopuli)

**Sonuç:** değerlendirme kaybı 0.4795 (adım 1000) → 0.4656 → 0.4611 → **0.4596** (adım 4000).
Ödev bir metrik eşiği istemiyor; Hub'da `text-to-speech` etiketli geçerli bir model yeterli.
Anlaşılırlık kontrolü: üretilen örnekleri `whisper-base` (Felemenkçe) geri yazdı:

| Girdi metni | Whisper'ın duyduğu |
|---|---|
| hallo allemaal, ik praat nederlands. groetjes aan iedereen! | Hallo allemaal. Ik ben het Nederlands, goedjes aan iedereen. |
| het europees parlement vergadert vandaag in straatsburg over het nieuwe klimaatbeleid. | Het opeest parlement, we gaan het vandaag in Stadsburg over het nieuwe klimaatbedrijf. |

Ses dosyaları: `results/unit6_sample_nl_s4000_*.wav` (GitHub'a yüklenmez; yerelde dinleyebilirsin).

Kod: [`unit6_speecht5_tts/src/prepare.py`](../unit6_speecht5_tts/src/prepare.py),
[`train.py`](../unit6_speecht5_tts/src/train.py), [`publish.py`](../unit6_speecht5_tts/src/publish.py).

## 1. Kod ne yapıyor?

| Adım | Nerede | Ne yapıyor |
|---|---|---|
| Veri | `prepare.main` | `facebook/voxpopuli` `nl` eğitim seti (Avrupa Parlamentosu konuşmaları, 20 968 kayıt, 84 konuşmacı). Kurs sayfası artık `qmeeus/voxpopuli` gösteriyor ama orada konuşmacı kimliği yok; konuşmacı filtresi için orijinal set gerekli. |
| Konuşmacı filtresi | | 100–400 kaydı olan 42 konuşmacı → 9 973 kayıt. Çok az kaydı olan konuşmacıdan ses kimliği öğrenilemez, çok fazlası seti tek sese boğar. |
| Metin temizleme | `cleanup` | SpeechT5 tokenizer'ı İngilizce karakter tabanlı: à ç è ë í ï ö ü → a c e e i i o u. Sonra bilinmeyen karakter sayısı kontrol edildi: **0**. Sayılar zaten `normalized_text`'te yazıyla. |
| Hedef spektrogram | `processor(audio_target=...)` | 16 kHz ses → 80 kutulu log-mel (decoder'ın tahmin edeceği hedef). |
| Konuşmacı vektörü | `speechbrain/spkrec-xvect-voxceleb` | Her kayıt için 512 boyutlu **x-vector**, L2 normalize. Model "kimin sesiyle" konuşacağını bundan öğrenir. |
| Uzunluk filtresi | | ≥ 200 token olan girdiler atılır → 8 259; %90/%10 → 7 433 eğitim, 826 test. |
| Collator | `TTSDataCollatorWithPadding` | Spektrogram dolgusu → −100 (kayıpta yok sayılır). Hedef uzunluk **reduction factor = 2**'nin katına yuvarlanır (decoder her adımda 2 kare üretir). x-vector'lar float32'ye çevrilir (float64 → fp16 matris çarpımında hata veriyordu). |
| Eğitim | `Seq2SeqTrainer` | lr 1e-5, warmup 500, 4000 adım, batch 4 × accumulation 8 = 32, gradient checkpointing, fp16, `use_cache=False` (checkpointing ile uyumsuz; üretimde tekrar açılır). ~1.2 sn/adım, ~85 dk. |
| Örnek üretimi | `generate_speech(..., vocoder=HiFi-GAN)` | Metin + test setinden bir x-vector → log-mel → `microsoft/speecht5_hifigan` → 16 kHz dalga formu. |

## 2. Matematik

### 2.1 SpeechT5 kaybı
Decoder her adımda bir mel karesi (aslında 2 kare) ve bir **durma (stop) olasılığı** üretir. Toplam kayıp:

L = L1(mel_önce_postnet, hedef) + L1(mel_sonra_postnet, hedef) + BCE(durma) + L_rehberli_dikkat

- **L1 (mutlak hata):** |tahmin − hedef| ortalaması. L2'ye göre aykırı karelere daha az duyarlı, spektrogramı daha az bulanık yapar.
  Post-net, kaba mel tahminini düzelten küçük bir konvolüsyon ağıdır; her iki çıktıya da kayıp uygulanır.
- **BCE (ikili çapraz entropi) — durma token'ı:** her kare için "konuşma burada bitti mi?" sorusu.
  L = −[y·log σ(z) + (1−y)·log(1−σ(z))]. Bir cümlede yüzlerce "devam" karesine karşı tek bir "dur" karesi var;
  bu dengesizliği dengelemek için pozitif sınıfa **ağırlık 5** veriliyor (`pos_weight=5`). Bu öğrenilmezse model susmaz ya da erken keser.
- **Rehberli dikkat (guided attention):** metin ve ses kabaca doğrusal hizalanır (metnin başı sesin başında). Dikkat
  matrisinin köşegenden uzak kısmına ceza vererek hizalamayı hızlı öğretir.

Değerlendirme kaybı 0.4795'ten 0.4596'ya indi ve 3000. adımdan sonra neredeyse düzleşti → daha uzun eğitimin getirisi az.

### 2.2 x-vector (konuşmacı gömmesi)
Bir TDNN (zaman gecikmeli sinir ağı) konuşmacı tanıma için eğitilir; ara katmanın zamansal ortalaması + standart sapması
(istatistik havuzlama) 512 boyutlu sabit bir vektör verir. Aynı kişinin farklı cümleleri birbirine yakın, farklı kişiler
uzak düşer. TTS bu vektörü decoder'a ekleyerek ses tınısını koşullar.

### 2.3 HiFi-GAN vocoder — GAN kayıpları
Mel spektrogram fazı içermez; dalga formunu yeniden kurmak için HiFi-GAN kullanılır. Eğitimi (Microsoft tarafından yapıldı,
biz sadece kullanıyoruz):

- Üretici G: mel → dalga formu. Ayırt ediciler D: çok periyotlu + çok ölçekli (gerçek/sahte ses).
- **LSGAN kaybı:** L_D = E[(D(x) − 1)²] + E[D(G(s))²],  L_G,adv = E[(D(G(s)) − 1)²]
- **Feature matching:** gerçek ve sahte sesin D'nin ara katmanlarındaki özelliklerinin L1 farkı.
- **Mel kaybı:** üretilen sesin mel'i ile gerçek mel'in L1 farkı (×45 ağırlıkla).

## 3. Önceki projelerinle bağlantı

**cGAN projesi** (`conditional-gan-animal-faces`) ile doğrudan paralel:

| | cDCGAN (senin) | SpeechT5 + HiFi-GAN |
|---|---|---|
| Koşul | sınıf etiketi → `nn.Embedding(3, 32)` | metin token'ları + 512-d x-vector |
| Üretici çıktısı | 64×64 RGB yüz | 80 kutulu mel → (HiFi-GAN) dalga formu |
| Kayıp | ikili çapraz entropi (doymayan GAN kaybı) | SpeechT5: L1 + BCE; HiFi-GAN: LSGAN + feature matching + mel L1 |
| Kararlılık hilesi | one-sided label smoothing | LSGAN (kare hata, doymaz) + yeniden oluşturma kaybı |
| Değerlendirme | FID 171.11 (Inception özellikleri) | eval L1 kaybı + ASR ile anlaşılırlık |

Önemli fark: SpeechT5'in kendisi bir GAN değil; mel'i doğrudan regresyonla (L1) öğrenir. GAN kısmı yalnızca vocoder'da.
Senin cGAN'ında ayırt edicinin rolünü burada iki şey üstleniyor: L1 kaybı (spektrogram yakınlığı) ve vocoder'ın ayırt edicileri (gerçekçi dalga formu).

**AST projesi:** x-vector çıkaran ağ, AST'deki `[CLS]` token'ının yaptığı işi yapar: değişken uzunluktaki sesi sabit
boyutlu tek bir vektöre özetlemek. AST bu vektörü sınıflandırmaya, SpeechT5 ise üretimi koşullamaya kullanır.
