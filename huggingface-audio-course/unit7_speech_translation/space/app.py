import re

import gradio as gr
import librosa
import numpy as np
import torch

from transformers import MarianMTModel, MarianTokenizer, VitsModel, VitsTokenizer, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1) speech (any language) -> English text: Whisper's built-in "translate" task
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# 2) English text -> Turkish text: Marian NMT model trained on OPUS / Tatoeba
mt_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr")
mt_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-tr").to(device)

# 3) Turkish text -> Turkish speech: MMS-TTS (VITS) for Turkish
tts_model = VitsModel.from_pretrained("facebook/mms-tts-tur").to(device)
tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-tur")
TTS_VOCAB = set(tts_tokenizer.get_vocab())


def translate(audio):
    # Decode + resample to 16 kHz ourselves (no dependency on an ffmpeg binary).
    waveform, _ = librosa.load(audio, sr=16000, mono=True)
    outputs = asr_pipe(
        {"raw": waveform, "sampling_rate": 16000}, max_new_tokens=256, generate_kwargs={"task": "translate"}
    )
    english_text = outputs["text"].strip()
    if not english_text:
        return ""
    # Translate sentence by sentence so long inputs stay within the MT model's context.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", english_text) if s]
    batch = mt_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generated = mt_model.generate(**batch, num_beams=4, max_new_tokens=256)
    turkish_text = " ".join(mt_tokenizer.batch_decode(generated, skip_special_tokens=True))
    print(f"EN: {english_text} | TR: {turkish_text}", flush=True)
    return turkish_text


def normalise_turkish(text):
    # Turkish-aware lower-casing (I -> ı, İ -> i), then keep only characters the MMS vocabulary knows.
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = "".join(c if c in TTS_VOCAB else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def synthesise(text):
    text = normalise_turkish(text)
    if not text:
        return torch.zeros(tts_model.config.sampling_rate // 2)
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model(inputs["input_ids"].to(device)).waveform[0]
    return speech.cpu()


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * 32767).astype(np.int16)
    return 16000, synthesised_speech


title = "Cascaded STST: any language to Turkish"
description = """
Cascaded speech-to-speech translation (STST) from source speech in any language to target speech in **Turkish**:

1. [Whisper Base](https://huggingface.co/openai/whisper-base) translates the speech into English text,
2. [opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) translates English text into Turkish,
3. [MMS-TTS Turkish](https://huggingface.co/facebook/mms-tts-tur) synthesises Turkish speech (16 kHz).

Hands-on exercise for Unit 7 of the Hugging Face Audio Course.

![Cascaded STST](https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png "Diagram of cascaded speech to speech translation")
"""

demo = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    examples=[["./example.wav"]],
    title=title,
    description=description,
)

demo.launch()
