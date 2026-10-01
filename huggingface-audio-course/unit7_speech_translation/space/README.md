---
title: Speech To Speech Translation Turkish
emoji: 🏆
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
python_version: '3.11'
app_file: app.py
pinned: false
license: mit
short_description: Any language speech to Turkish speech (Whisper, Marian, MMS)
---

# Cascaded speech-to-speech translation: any language -> Turkish

Unit 7 hands-on exercise of the [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter7/hands_on),
based on the template Space [`course-demos/speech-to-speech-translation`](https://huggingface.co/spaces/course-demos/speech-to-speech-translation).
The `speech_to_speech_translation(audio) -> (16000, int16 array)` signature is unchanged.

Pipeline (runs on the free CPU tier):

| Step | Model | Output |
|---|---|---|
| Speech translation | [openai/whisper-base](https://huggingface.co/openai/whisper-base) (`task="translate"`) | English text |
| Text translation | [Helsinki-NLP/opus-mt-tc-big-en-tr](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-tr) | Turkish text |
| Text-to-speech | [facebook/mms-tts-tur](https://huggingface.co/facebook/mms-tts-tur) (VITS) | Turkish speech, 16 kHz |

Source code: [github.com/emrezorlu1239/course-projects/tree/main/huggingface-audio-course](https://github.com/emrezorlu1239/course-projects/tree/main/huggingface-audio-course)
