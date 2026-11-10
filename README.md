# AI Debate Arena

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Inference-Ollama-000000)
![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-1C3C3C)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

AI Debate Arena runs a structured debate between two local AI agents. One argues for a proposition, the other argues against it, and an impartial judge evaluates the complete transcript. It is a small, readable example of coordinating local Ollama models with LangGraph.

All inference is performed through your configured Ollama service—there is no hosted model API key or cloud model requirement.

## Features

- Two debaters with fixed, opposing positions
- Multi-round arguments that respond to the existing transcript
- An impartial judge that evaluates reasoning, relevance, evidence, and responsiveness
- Scores for both sides, identified gaps, and a final verdict
- Interactive CLI or fully scriptable command-line options
- Friendly guidance when Ollama is unavailable or a model has not been downloaded

## Debate workflow

```text
Topic
  │
  ▼
Debater For ──► Debater Against
  ▲                    │
  │       more rounds? │ no
  └────── yes ─────────┘
                       ▼
                 Impartial Judge
                       │
                       ▼
                 Final verdict
```

LangGraph keeps the topic, current round, full argument list, and verdict in one shared state object. The judge sees the full transcript only after both agents complete the selected number of rounds.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- A pulled chat model, such as `llama3.2`

## Quick start

Clone the repository and enter it:

```bash
git clone https://github.com/tabishsyed777/AI-debate-arena-by-tabz.git
cd AI-debate-arena-by-tabz
```

Install the Python dependencies and download a model:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2
```

Start Ollama before running the app: open the Ollama desktop app, or run `ollama serve` in a separate terminal.

Then start an interactive debate:

```bash
python main.py
```

Example session:

```text
Debate topic: Should remote work remain a permanent option?
Number of rounds [2]: 2
```

## Command-line use

Provide every option for scripts or repeatable runs:

```bash
python main.py \
  --topic "Should AI replace traditional classroom learning?" \
  --rounds 3 \
  --model llama3.2
```

| Option | Description | Default |
| --- | --- | --- |
| `--topic` | Proposition for the agents to debate | Prompted interactively |
| `--rounds` | Number of rounds, from 1 to 10 | Prompted; 2 |
| `--model` | Locally installed Ollama model | `llama3.2` |
| `--ollama-url` | Custom Ollama endpoint | Ollama default |

For example, point the app at a remote or non-default Ollama server:

```bash
python main.py --topic "Should cities ban private cars downtown?" --rounds 2 \
  --ollama-url http://localhost:11434
```

## What the judge returns

The judge is asked to return a Markdown report with:

- The strongest arguments from both positions
- Logical gaps or unsupported claims
- A score out of 10 for each debater
- A verdict naming a winner or a tie, with an explanation

The application displays a formatting note if a model does not include every requested report section.

## Project structure

```text
.
├── main.py           # CLI, prompts, LangGraph state, and workflow
├── requirements.txt  # Python dependencies
├── LICENSE           # MIT License
└── README.md         # Documentation
```

## Troubleshooting

**“Could not run the debate”**  
Ensure Ollama is running (`ollama serve`) and that the requested model is installed (`ollama pull llama3.2`).

**Model name not found**  
List downloaded models with `ollama list`, then pass one with `--model`.

**Connection to a custom server fails**  
Confirm that `--ollama-url` includes the scheme and port, for example `http://localhost:11434`.

## Contributing

Contributions are welcome. Keep changes focused, preserve the local-first design, and test against a running Ollama instance when changing model or graph behavior.

## License

This project is released under the [MIT License](LICENSE).
