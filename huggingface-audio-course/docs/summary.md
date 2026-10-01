# Özet — dört ödev, tek resim

| Ünite | Görev | Model | Kayıp (eğitim) | Metrik (sonuç) |
|---|---|---|---|---|
| 4 | ses → sınıf | AST (Transformer encoder) | çapraz entropi | doğruluk 0.8978 |
| 5 | ses → metin | Whisper (encoder–decoder) | token başına çapraz entropi | normalize WER 0.2940 |
| 6 | metin → ses | SpeechT5 + HiFi-GAN | mel L1 + durma BCE (+ rehberli dikkat); vocoder'da GAN | eval kaybı 0.4596 |
| 7 | ses → ses (başka dil) | Whisper → Marian → VITS | (hazır modeller) | dil tanıma `tur` 0.99997 |

## Ortak omurga: log-mel spektrogram
Dört ünitenin hepsi sesi aynı ara temsile çeviriyor: kısa pencerelerde Fourier → mel ölçeği → logaritma.
- AST: 128 mel × 1024 kare (10 sn) → **girdi**
- Whisper: 80 mel × 3000 kare (30 sn) → **girdi**
- SpeechT5: 80 mel → **çıktı** (hedef); HiFi-GAN bunu dalga formuna çevirir
- MMS-TTS (VITS): mel'i atlayıp doğrudan dalga formu üretir (ama eğitiminde mel L1 kaybı var)

Yani sınıflandırma ve tanımada spektrogram **okunan**, sentezde **üretilen** şey.

## Kayıp fonksiyonları — tek tabloda

| Kayıp | Formül (sade) | Nerede |
|---|---|---|
| Çapraz entropi | −log p(doğru sınıf) | AST (10 sınıf), Whisper (her token ~51k sınıf), cGAN ayırt edicisi (2 sınıf) |
| L1 | ortalama \|tahmin − hedef\| | SpeechT5 mel, HiFi-GAN mel, VITS yeniden oluşturma |
| İkili çapraz entropi (ağırlıklı) | −[5·y log σ(z) + (1−y) log(1−σ(z))] | SpeechT5 durma token'ı |
| LSGAN | D: (D(x)−1)² + D(G)²; G: (D(G)−1)² | HiFi-GAN, VITS |
| Doymayan GAN kaybı | G: −log D(G(z)) | senin cDCGAN'ın |
| KL ıraksaması | iki dağılım arası "uzaklık" | VITS (VAE kısmı) |

## Metrikler

| Metrik | Ne ölçer | Not |
|---|---|---|
| Doğruluk | doğru tahmin oranı | n = 150 klipte SE ≈ 0.023 → eşiğin hemen üstü güvenli değil |
| WER | (S + D + I) / N, Levenshtein ile | 1'i geçebilir; normalizasyon sonucu değiştirir |
| Eval kaybı (TTS) | L1 + BCE + dikkat | düşük kayıp ≠ doğal ses; dinleyerek + ASR ile kontrol ettik |
| Dil tanıma skoru | MMS-LID softmax olasılığı | Ünite 7 değerlendiricisi ≥ 0.5 ve `eng` değil ister |
| FID (cGAN) | Inception özellik dağılımları arası Fréchet uzaklığı | senin cGAN'ında 171.11 |

## Önceki projelerinle bağ (özet)

- **AST hayvan sesleri → Ünite 4:** aynı ön-eğitimli ağırlık, aynı 1024×128 girdi, aynı SpecAugment fikri, aynı
  "sızıntısız bölme" disiplini. Fark: `Trainer` ve 10 sn'lik parçalar; sonuç %86.25 → %89.8.
- **cGAN → Ünite 6 ve 7:** koşullu üretim. cGAN'da koşul sınıf etiketiydi; SpeechT5'te metin + x-vector, VITS'te metin.
  GAN fikri aynen HiFi-GAN ve VITS'in dalga formu üreticisinde yaşıyor.
- **Hepsinde:** transfer öğrenme. Hiçbir ünitede sıfırdan eğitim yok; büyük ön-eğitimli modeli küçük bir veri setine
  (800 klip, 450 cümle, 7 433 cümle) uyarladık. AST projesindeki "sıfırdan %32.5 vs transfer %86.25" sonucu bunun neden şart olduğunu gösteriyor.

## Bu projede karşılaşılan gerçek sorunlar (ve dersler)

1. `datasets` 5.x: script tabanlı veri setleri ve torchcodec'siz ses çözme yok → parquet + ham bayt + soundfile.
2. transformers 5.18: `load_best_model_at_end` AST'de sessizce başarısız → en iyi checkpoint'i elle yükle, **son metriği doğrula**.
3. Kurs sayfası `qmeeus/voxpopuli`'ye geçmiş ama konuşmacı kimliği yok → orijinal `facebook/voxpopuli`.
4. float64 x-vector + fp16 → dtype hatası → float32'ye çevir.
5. `pipeline("translation")` kaldırılmış; Windows'ta ffmpeg ile çözme başarısız → Marian'ı doğrudan çağır, librosa ile yükle.
6. Değerlendiricinin kodunu okumak işe yaradı: WER'in ondalık olması, `datasets` metadatasının birebir eşleşmesi ve
   Ünite 7'nin ayrıca bir değerlendirme Space'ine gönderilmesi gerektiği oradan anlaşıldı.
