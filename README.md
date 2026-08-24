# Agentic AI Chatbot

An educational project for learning how modern agentic systems are assembled with
LangGraph, LangChain, Google Gemini, LangSmith, SQLite, and local retrieval.

Phase 1 is a minimal Gemini chatbot built as an explicit LangGraph graph. The
terminal keeps conversation messages in memory while the program runs.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

Create the environment and install the locked dependencies:

```bash
uv sync --dev
```

Copy the environment template:

```bash
cp .env.example .env
```

Add your Google AI Studio API key to `.env`:

```dotenv
GOOGLE_API_KEY=your-google-api-key
```

Do not commit `.env`; it is ignored by Git.

## Run

```bash
uv run agentic-chatbot
```

The prompt will appear after startup:

```text
Gemini chatbot ready. Type 'exit' or 'quit' to stop.
You:
```

You can also run the package as a module:

```bash
uv run python -m agentic_chatbot
```

## Test

```bash
uv run pytest
```

## Project layout

```text
src/agentic_chatbot/
├── __init__.py
├── __main__.py
├── cli.py
├── config.py
├── graph.py
├── logging_config.py
├── model.py
└── main.py
tests/
├── test_cli.py
├── test_config.py
├── test_graph.py
└── test_main.py
```

## Graph architecture

```text
START ──→ gemini node ──→ END
              │
              ├─ reads all messages from MessagesState
              ├─ sends them to ChatGoogleGenerativeAI
              └─ returns the new AI message to be appended to state
```

The graph is assembled directly with `StateGraph`, `MessagesState`, `START`, and
`END`. No prebuilt agent abstraction is used.

## Configuration

Configuration is read from environment variables and, when present, a local
`.env` file. See `.env.example` for all supported values. Sensitive values are
represented as secret values and are never written to startup logs.

`GEMINI_MODEL` selects the model and defaults to `gemini-2.5-flash`.

### LangSmith tracing

To inspect graph and model runs in LangSmith, update `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentic-chatbot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

If the API key can access multiple workspaces, also set
`LANGSMITH_WORKSPACE_ID`. With tracing enabled, conversation inputs, outputs,
timings, and execution metadata are sent to LangSmith. Keep tracing disabled for
content you do not want recorded there.

## Try a conversation

```text
You: My name is Aziz.
Assistant: Nice to meet you, Aziz!
You: What is my name?
Assistant: Your name is Aziz.
You: quit
Goodbye!
```

Conversation history is maintained only by the current CLI process. Exiting the
program loses that history because persistence is intentionally deferred.

Tools, persistence, human approval, retrieval, and UI layers remain future phases.
