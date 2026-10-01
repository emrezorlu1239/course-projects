# Course Projects

Hands-on projects and assignments from the AI / machine learning courses I have completed, collected in one place. Each course lives in its own folder with its notebooks, results and certificate.

| Course | Provider | Topics | Result | Link |
|---|---|---|---|---|
| Deep Reinforcement Learning Course | Hugging Face | Q-learning, DQN, REINFORCE, A2C, PPO (from scratch), curiosity (RND), multi-agent self-play (MA-POCA), APPO | 11 / 11 assignments passed · **Certificate of Excellence** | [`huggingface-deep-rl-course`](huggingface-deep-rl-course) |
| AI Agents Course | Hugging Face | Tool-calling agents, function calling, GAIA-style research agent (Atlas), local LLM inference | Unit 1: 100% · Final: 35% (pass ≥ 30%) · **Certificate of Excellence** | [`huggingface-agents-course`](huggingface-agents-course) |
| Audio Course | Hugging Face | Audio classification (AST), speech recognition (Whisper), text-to-speech (SpeechT5), speech-to-speech translation | Units 4–6 passed on the Hub (GTZAN 0.898, WER 0.294, Dutch TTS) · Unit 7 Space and certificate pending | [`huggingface-audio-course`](huggingface-audio-course) |

## Structure

```
course-projects/
├── huggingface-deep-rl-course/
│   ├── README.md              # results, engineering notes, key concepts
│   ├── notebooks/             # one notebook per assignment
│   └── certificate/
├── huggingface-agents-course/
│   ├── README.md              # results and final project overview
│   ├── atlas-research-agent/  # Unit 4 final project
│   └── certificate/
└── huggingface-audio-course/
    ├── README.md              # results, Hub links, setup
    ├── unit4_… unit7_…/       # AST, Whisper, SpeechT5, speech translation
    ├── results/ docs/
    └── requirements.txt
```

New courses are added as sibling folders.
