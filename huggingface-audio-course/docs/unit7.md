# Ünite 7 — Konuşmadan konuşmaya çeviri (herhangi bir dil → Türkçe)

**Sonuç (yerel test):** değerlendiricinin kendi dosyası `test_short.wav` → çıktı sesi `facebook/mms-lid-126` tarafından
**`tur`** olarak tanındı (skor 0.99997; eşik ≥ 0.5 ve `eng` olmamalı). CPU'da gecikme 7.9–17.8 sn.

Kod: [`unit7_speech_translation/space/app.py`](../unit7_speech_translation/space/app.py) (Space'in kendisi),
[`unit7_speech_translation/src/check_space.py`](../unit7_speech_translation/src/check_space.py) (değerlendiricinin yerel kopyası).

## 1. Kod ne yapıyor?

Kurs şablonu `course-demos/speech-to-speech-translation` → ses → Whisper (İngilizceye çeviri) → SpeechT5 (İngilizce ses).
Ödev: çıktı İngilizce **olmamalı** ve `speech_to_speech_translation(audio) -> (16000, int16 dizi)` imzası korunmalı.

Kademeli (cascaded) zincirimiz:

| Adım | Model | Girdi → çıktı |
|---|---|---|
| 1. Konuşma çevirisi | `openai/whisper-base` (`task="translate"`) | herhangi bir dilde ses → İngilizce metin |
| 2. Metin çevirisi | `Helsinki-NLP/opus-mt-tc-big-en-tr` (Marian) | İngilizce → Türkçe metin (cümle cümle, beam search 4) |
| 3. Metinden sese | `facebook/mms-tts-tur` (VITS) | Türkçe metin → 16 kHz Türkçe ses |

Ayrıntılar:
- **Ses okuma:** `librosa.load(audio, sr=16000)` — ffmpeg ikili dosyasına bağımlı değiliz (yerelde ffmpeg ile çözme başarısız oldu).
- **transformers 5:** `pipeline("translation")` kaldırıldığı için Marian modeli doğrudan `generate()` ile çağrılıyor.
- **Türkçe normalizasyon:** MMS sözlüğü küçük harf; Python'un `"I".lower()` sonucu "i" (Türkçede "ı" olmalı), bu yüzden
  önce `I→ı`, `İ→i`, sonra küçük harf; sözlükte olmayan karakterler (noktalama, bazı rakamlar) boşluğa çevrilir.
- **Çıktı:** float dalga formu × 32767 → int16, örnekleme hızı 16000 (şablonla aynı).
- **Değerlendirici neyi kontrol ediyor?** Space'i `gradio_client` ile çağırır (`/predict`, `test_short.wav`), dönen
  sesi MMS-LID-126 ile dil tanımaya sokar. `check_space.py` bunu birebir taklit eder; Space, HF Space'teki ile aynı
  sürümlerle kurulmuş ayrı bir venv'de (`.venv-space`: gradio 5.49.1, transformers 5.18, CPU torch) test edildi.

## 2. Matematik

### 2.1 Kademeli sistemde hata birikimi
Her aşama hatasız değil. Basit bir model: üç aşamanın her biri cümleyi p₁, p₂, p₃ olasılıkla doğru geçiriyorsa,
uçtan uca doğru olma olasılığı yaklaşık p₁·p₂·p₃'tür (ör. 0.9 · 0.9 · 0.95 ≈ 0.77). Bu yüzden her aşamada mümkün olan
en iyi modeli seçtik (tc-big MT modeli, beam search). Uçtan uca (direct) S2ST modelleri bu birikimi önler ama çok daha
fazla eşleşmiş ses verisi ister.

### 2.2 Beam search
Otoregresif decoder her adımda olası sonraki token'lara olasılık verir. Açgözlü (greedy) arama her adımda en olasıyı
seçer; beam search ise en iyi **k = 4** kısmi cümleyi paralel tutar ve toplam log-olasılığı Σ log p(y_t | y_<t, x)
en yüksek olanı seçer. Bir adımda daha az olası ama sonrasında çok daha akıcı devam eden çevirileri kaçırmaz.

### 2.3 VITS (MMS-TTS) — bir GAN!
VITS üç parçayı birlikte eğitir:
1. **Değişken otokodlayıcı (VAE):** ses → gizli değişken z → ses. Kayıp: yeniden oluşturma (mel L1) + KL ıraksaması
   (z'nin dağılımı ile metinden tahmin edilen önsel dağılım arasındaki uzaklık).
2. **Normalizing flow:** metin tarafındaki basit dağılımı ses tarafındaki karmaşık dağılıma tersinir dönüşümlerle eşler.
3. **Çekişmeli (adversarial) eğitim:** dalga formunu üreten decoder (HiFi-GAN üreticisi) bir **ayırt edici**ye karşı eğitilir
   (least-squares GAN kaybı + feature matching).

Ayrıca **stokastik süre tahmincisi** her harfin kaç kare süreceğini örnekler; bu yüzden aynı cümle her seferinde biraz
farklı ritimle okunabilir.

## 3. Önceki projelerinle bağlantı

- **cGAN projesi:** VITS'in ses üreticisi, senin cDCGAN'ındaki üretici–ayırt edici oyununun aynısını oynar. Fark:
  senin modelinde koşul bir **sınıf etiketi** (kedi/köpek/vahşi) idi; VITS'te koşul **metin** (ve süre). cGAN'da
  kararlılık için one-sided label smoothing ve doymayan kayıp kullandın; VITS ise LSGAN kaybı + feature matching + VAE
  yeniden oluşturma kaybı ile kararlı hâle gelir.
- **AST projesi:** zincirin "dinleyen" ucu (Whisper) ve değerlendiricinin dil tanıma modeli (MMS-LID, wav2vec2 tabanlı
  bir sınıflandırıcı) — ikisi de senin AST projesindeki gibi ses → sınıf/metin eşlemesi yapar. MMS-LID tam olarak
  AST'nin yaptığı işi (ses sınıflandırma) 126 dil sınıfıyla yapıyor.
