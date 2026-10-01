# Ünite 5 — Whisper-tiny ince ayarı (MINDS-14 en-US)

**Sonuç:** normalize WER **0.2940** (ortografik 0.2887). İnce ayarsız whisper-tiny aynı değerlendirme setinde 0.4014.
Ödev eşiği: normalize WER < 0.37 (ondalık).

Kod: [`unit5_whisper_finetune/src/train.py`](../unit5_whisper_finetune/src/train.py),
[`publish.py`](../unit5_whisper_finetune/src/publish.py).

## 1. Kod ne yapıyor?

| Adım | Nerede | Ne yapıyor |
|---|---|---|
| Veri | `load_splits` | `PolyAI/minds14`, `en-US` (563 kayıt, 8 kHz telefon sesi). İlk 450 eğitim, kalan 113 değerlendirme (ödevin istediği). `map(..., num_proc=1)`. |
| Ses → girdi | `prepare` | 16 kHz'e örnekle; `WhisperFeatureExtractor` 30 sn'ye dolgulayıp 80 kutulu log-mel çıkarır (3000 kare). >30 sn olan 5 kayıt eğitimden atılır (445 kalır). |
| Metin → etiket | `prepare` | Tokenizer `<|startoftranscript|><|en|><|transcribe|><|notimestamps|> metin <|endoftext|>` dizisini üretir. |
| Collator | `DataCollatorSpeechSeq2SeqWithPadding` | Etiketleri aynı uzunluğa dolgular, dolgu yerlerine **−100** koyar (kayıp onları yok sayar). Tüm satırlar `<|startoftranscript|>` ile başlıyorsa onu **keser**, çünkü model etiketleri sağa kaydırırken bu token'ı kendisi ekler; kesmezsek iki kez olur. |
| Metrik | `make_compute_metrics` | Tahmin ve referansı çöz; ortografik WER (ham metin) ve `BasicTextNormalizer` sonrası normalize WER. **×100 yapılmaz**, ondalık kalır. |
| Eğitim | `Seq2SeqTrainer` | lr 1e-5, warmup 50, 500 adım, batch 16, fp16, `predict_with_generate=True` (değerlendirmede gerçek otoregresif çözümleme), `generation_max_length=225`. Her 100 adımda değerlendirme. |
| En iyi checkpoint | | En düşük WER: adım 400. `from_pretrained` ile yüklenip son `evaluate()` yapılır; model kartındaki `- Wer: 0.2940` satırı buradan gelir. |

WER seyri: adım 0: 0.401 → 100: 0.302 → 200: 0.310 → 300: 0.316 → **400: 0.295** → 500: 0.296.
Değerlendirme kaybı 100. adımdan sonra artarken WER düşmeye devam etti: kayıp "emin olma" derecesini, WER ise sadece
argmax kelimeleri ölçer (Ünite 4'teki gözlemin aynısı).

## 2. Matematik

### 2.1 Encoder–decoder ve öğretmen zorlaması (teacher forcing)
Encoder log-mel'i okur, decoder metni token token üretir. Eğitimde decoder'a her adımda **gerçek** önceki token
verilir (teacher forcing) ve bir sonraki token için çapraz entropi hesaplanır:

L = − (1/T) Σ_t log p(y_t | y_<t, ses)

−100 olan konumlar bu toplamdan çıkarılır. Değerlendirmede ise gerçek token yok; model kendi ürettiğini geri besler
(`predict_with_generate=True`). WER'i doğru ölçmek için bu şart.

### 2.2 WER ve Levenshtein uzaklığı
WER = (S + D + I) / N  — S yer değiştirme, D silme, I ekleme, N referanstaki kelime sayısı.

S, D, I sayısı **Levenshtein (düzenleme) uzaklığı** ile bulunur. Dinamik programlama tablosu:

d[i][j] = min( d[i−1][j] + 1 (silme),  d[i][j−1] + 1 (ekleme),  d[i−1][j−1] + [r_i ≠ h_j] (eşleşme/yer değiştirme) )

Örnek: referans "i want to pay my bill", tahmin "i want pay my bills":
- "to" silindi (D=1), "bill"→"bills" (S=1) → WER = 2/6 = 0.33.

WER 1'i geçebilir (çok ekleme yaparsa), bu yüzden "doğruluk" değil, hata oranıdır.

### 2.3 Neden normalize WER?
`BasicTextNormalizer` küçük harfe çevirir, noktalama ve sembolleri siler. Böylece "Hello," ile "hello" aynı sayılır.
İlginç not: burada normalize WER (0.294) ortografikten (0.289) biraz yüksek çıktı. MINDS-14 metinleri zaten çoğunlukla
küçük harf ve noktalamasız; normalizer ise "I'm" gibi kesme işaretli kelimeleri "i m" diye **iki kelimeye** ayırıyor,
böylece tek bir hata iki hata sayılabiliyor.

### 2.4 CTC'den farkı
Wav2Vec2 gibi CTC modelleri her ses karesine bağımsız bir harf (veya boşluk) olasılığı verir ve hizalamayı
toplayarak öğrenir; dil modeli yoktur. Whisper ise decoder'ı sayesinde **koşullu bir dil modelidir**:
p(y_t | y_<t, ses). Bu yüzden yazım ve dilbilgisi daha iyi, ama tamamen uydurma ("halüsinasyon") da yapabilir.

## 3. Önceki projelerinle bağlantı

- **AST projesi:** orada da girdi bir log-mel spektrogramdı (128 kutu, 10 ms adım). Whisper 80 kutu ve 30 sn sabit
  pencere kullanır; encoder'ı yine bir Transformer, fark yama yerine 1-B konvolüsyon + konumsal kodlama kullanması.
- **cGAN projesi:** cGAN üreticisi "sınıf etiketi → görüntü" koşullu üretimi yapıyordu. Whisper'ın decoder'ı da
  koşullu bir üreticidir: "ses → metin". Fark: GAN bir ayırt edici ile dolaylı bir kayıp öğrenir, Whisper ise doğrudan
  log-olabilirlik (çapraz entropi) ile eğitilir — o yüzden eğitimi GAN'a göre çok daha kararlıdır.
