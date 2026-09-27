# Atlas Research Agent

A tool-calling research agent built for the **Hugging Face Agents Course, Unit 4 final assignment**.
It answers GAIA-style research questions by choosing tools, reading real sources, and returning a
single concise answer. It contains **no answer lookup table** and no benchmark answer bank.

## Published page and local demo

The Hugging Face Space is a **static project page**, not an interactive hosted agent.
Run the included Gradio app locally to ask questions. Local LM Studio mode needs no HF token;
cloud inference mode uses your Hugging Face token and account credits.

## Official evaluation result

Submitted to `POST https://agents-course-unit4-scoring.hf.space/submit` on 2026-09-17.

| | |
| --- | --- |
| **Score** | **35.0%** — 7 / 20 correct |
| Pass threshold | 30% |
| Model | `qwen3.5-9b` (Q4_K_M) served locally by LM Studio on a single 8 GB laptop GPU |
| Answers produced | 11 of 20; the remaining 9 were returned as `UNKNOWN` rather than guessed |
| Run SHA-256 | `23a28cad625f256491848d009d59ba086a4cec5210b43396aada9cd3cf86fa99` |

The API replied: *"Score calculated successfully: 7/20 total questions answered correctly (20 valid tasks attempted). High score updated on leaderboard."*

The score was produced by running this code against the twenty official questions. No GAIA answer
list, leaderboard scrape or `task_id → answer` table was used. `evaluation-summary.json` in this
repository records every question, the answer that was submitted, and which tools ran.

<details>
<summary>What the agent submitted for each question</summary>

| # | Question | Submitted answer |
| --- | --- | --- |
| 1 | How many studio albums were published by Mercedes Sosa between 2000 and … | `3` |
| 2 | In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the hi… | `UNKNOWN` |
| 3 | .rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnat… | `right` |
| 4 | Review the chess position provided in the image. It is black's turn. Pro… | `UNKNOWN` |
| 5 | Who nominated the only Featured Article on English Wikipedia about a din… | `FunkMonk` |
| 6 | Given this table defining * on the set S = {a, b, c, d, e}  /*/a/b/c/d/e… | `b,e` |
| 7 | Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.  What … | `UNKNOWN` |
| 8 | What is the surname of the equine veterinarian mentioned in 1.E Exercise… | `UNKNOWN` |
| 9 | I'm making a grocery list for my mom, but she's a professor of botany an… | `broccoli, celery, fresh basil, lettuce, sweet potatoes` |
| 10 | Hi, I'm making a pie but I could use some help with my shopping list. I … | `UNKNOWN` |
| 11 | Who did the actor who played Ray in the Polish-language version of Every… | `Wojciech` |
| 12 | What is the final numeric output from the attached Python code? | `UNKNOWN` |
| 13 | How many at bats did the Yankee with the most walks in the 1977 regular … | `519` |
| 14 | Hi, I was out sick from my classes on Friday, so I'm trying to figure ou… | `UNKNOWN` |
| 15 | On June 6, 2023, an article by Carolyn Collins Petersen was published in… | `UNKNOWN` |
| 16 | Where were the Vietnamese specimens described by Kuznetzov in Nedoshivin… | `St. Petersburg` |
| 17 | What country had the least number of athletes at the 1928 Summer Olympic… | `CUB` |
| 18 | Who are the pitchers with the number before and after Taishō Tamai's num… | `Yamasaki, Uehara` |
| 19 | The attached Excel file contains the sales of menu items for a local fas… | `UNKNOWN` |
| 20 | What is the first name of the only Malko Competition recipient from the … | `Mikhail` |

</details>

### Certificate

The Hugging Face Agents Course *Certificate of Excellence* was issued on 2026-09-17 once this score
was recorded on the course leaderboard:

<img src="../certificate/certificate_of_excellence.png" alt="Certificate of Excellence" width="480">

See the [course overview](../README.md) for the Unit 1 certificate as well.

## How it works

```
question ─► language model ─► tool call(s) ─► observations ─► ... ─► final answer
                  ▲                                  │
                  └──────────── conversation ────────┘
```

`agent.py` runs a provider-independent OpenAI-style function-calling loop:

1. The system prompt states the task, the evidence rules, and the required answer format.
2. The model picks tools from an explicit registry (`atlas_tools.REGISTRY`).
3. Each observation is appended to the conversation and the loop continues, up to `max_steps`.
4. A finalization pass reduces a prose reply to the bare answer the grader compares.

Design decisions that came out of measured failures, not guesses:

| Problem observed in a real run | General fix in the code |
| --- | --- |
| The model repeated the same `read_url` call 14 times and ran out of steps | Identical `(tool, arguments)` calls are detected and answered with a note instead of being re-executed |
| One step emitted 15 tool calls and flooded the context | `calls_per_step` caps executed calls; the rest are answered with a budget note |
| Long pages evicted earlier observations, so the model forgot what it had read | `compact()` shrinks the oldest observations to keep the chain inside the context window |
| The model returned "the studio albums are ... thus the number is 3" | `looks_bare()` + a finalization prompt return the value only |
| A question asking for "X, Y" came back as "X Y" | The finalization pass is forced when the question requests a separated form |
| A legacy Windows code page crashed the run on a Japanese name | Progress logging encodes defensively |

### Tools

| Tool | Purpose |
| --- | --- |
| `search_web` | Public web search (`ddgs`), with retries, excluding benchmark answer datasets |
| `read_url` | Fetch an HTML page or PDF, extract text and links, optionally return only passages around a phrase |
| `read_attachment` | Read a task attachment: spreadsheet, PDF, text or source code. Source code is **read, never executed** |
| `calculate` | Arithmetic over a restricted AST (`+ - * / // %`, `sum/min/max/round/abs/len`). Not `eval` |
| `reverse_text` | Character reversal, for reversed or encoded prompts |

### Safety properties

- `public_url()` rejects non-HTTP(S) schemes and any host resolving to a private, loopback or
  link-local address, including on redirects. This blocks SSRF against cloud metadata endpoints.
- `calculate()` walks a whitelisted AST. `__import__('os').getcwd()`, `(1).__class__` and
  `2**999999` all raise. There is no `eval`, no `exec`, no shell.
- Downloads are capped at 8 MB; observations and the conversation are capped in characters.
- Retrieved web content is treated as untrusted data, never as instructions.
- The static Space runs no inference. The local Gradio app requires a token only in cloud mode; local requests never receive the HF token.

## Running it

Requires Python 3.12. Run the commands from this folder (`huggingface-agents-course/atlas-research-agent`).

```sh
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Hosted inference (Hugging Face Inference Providers)

```sh
export HF_TOKEN=...                 # never commit this
export ATLAS_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507:novita
python app.py
python evaluate.py --output work/evaluation.json
```

### Local inference, no cloud credits

Any OpenAI-compatible local server works. The recorded evaluation used
[LM Studio](https://lmstudio.ai/) serving `qwen3.5-9b` (Q4_K_M, 16384-token context) on a single
8 GB laptop GPU:

```sh
lms load qwen3.5-9b --context-length 16384 --gpu max
lms server start
python run_local.py --ui
# Or run a fresh evaluation with the recorded settings:
python run_local.py --output work/evaluation-local-new.json
```

`run_local.py` restores the recorded limits: 8 research steps, 3 calls per step,
7,000-character observations/documents, 34,000-character conversation budget,
500/1,300-character snippet windows, 5 snippets and 25 links. Environment overrides remain available.
The original GGUF download repository, revision and hash were not recorded; the known model label
is `qwen3.5-9b`, Q4_K_M. These settings do not guarantee identical answers or scores, because model
provenance is incomplete and web sources change. The historical 35% result remains unchanged.

`local_backend.py` is a second option: a **loopback-only** HTTP server that loads
a local Transformers checkpoint and translates the model's XML tool-call syntax into OpenAI
function-calling shape. It binds `127.0.0.1` only, has no authentication, and is not intended to be
exposed to a network.

```sh
pip install -r requirements-local.txt
python local_backend.py --snapshot /path/to/model/snapshot
export ATLAS_ENDPOINT=http://127.0.0.1:8766/v1/chat/completions
```

The optional Transformers backend requires a compatible NVIDIA CUDA GPU. Its dependencies are separate from the LM Studio path; installing the base requirements alone does not enable it.

### Tuning knobs

`ATLAS_ENDPOINT`, `ATLAS_MODEL`, `ATLAS_OBS_CHARS`, `ATLAS_CONTEXT_CHARS`, `ATLAS_DOC_CHARS`,
`ATLAS_SNIPPET_BEFORE`, `ATLAS_SNIPPET_AFTER`, `ATLAS_MAX_SNIPPETS`, `ATLAS_MAX_LINKS`.
Smaller local models need tighter document budgets than hosted ones.

### Submitting

```sh
python evaluate.py --output work/evaluation-local.json \
  --submit --username YOUR_USERNAME --space YOUR_USERNAME/atlas-research-agent
python evaluate.py --output work/evaluation-local.json --summary evaluation-summary.json
```

Submission writes a receipt next to the run file containing the SHA-256 of the exact run that was
submitted, so the score can be traced back to the traces that produced it.

## Limitations

These are real, measured limitations, not disclaimers.

- **No audio, video or image understanding.** The backend is text-only. Questions that require
  watching a video, hearing an MP3 or reading a chess position from a PNG are answered `UNKNOWN`.
  The agent is explicitly instructed never to claim it watched or heard something it did not.
- **The official scoring API served no attachments during this run.** Every
  `GET /files/{task_id}` returned `404 {"detail": "No file path associated with task_id ..."}`
  for all five file-backed questions, including the `.py` and `.xlsx` tasks that this agent's tools
  can otherwise handle. That is a server-side condition, not an agent failure, and it caps the
  reachable score.
- **Search is a free endpoint.** It rate-limits; `search_web` retries, but evidence can still be
  missed on a bad run.
- **A small local model is a real constraint.** A 9B model reasons less reliably over long
  multi-hop chains than a frontier model. An earlier run of this same code against a 4B local model
  answered far fewer questions; the traces for that run are kept.
- **A course score is not a general reliability guarantee.** It is one run, on twenty questions.
- The URL guard reduces SSRF risk but is not a substitute for network isolation in deployment.

## Repository layout

```
agent.py           tool-calling loop, context compaction, answer finalization
atlas_tools.py     the tool registry, URL guard and JSON schemas
evaluate.py        real evaluation runner, submission, and summary report
app.py             local Gradio UI; token needed only for cloud inference
run_local.py       recorded local evaluation settings and token-free UI launcher
requirements-local.txt optional CUDA Transformers dependencies
local_backend.py   optional loopback-only Transformers server
tests/             unit tests for the tool guards
```

## Provenance

Built by [Emre Zorlu](https://github.com/emrezorlu1239) for the Hugging Face Agents Course final
assignment. The tool loop, the research tools and the evaluation harness are implemented in this
repository; the assignment template and the course's function-calling guide were the starting point.

- https://huggingface.co/learn/agents-course/unit4/hands-on
- https://huggingface.co/docs/inference-providers/en/guides/function-calling

Licensed under the MIT License.
