# Agentic AI Chatbot

An educational project for learning how modern agentic systems are assembled with
LangGraph, LangChain, Google Gemini, LangSmith, SQLite, and local retrieval.

Phase 0 contains only the Python project foundation. It does not make model calls
or implement an agent yet.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)

## Setup

Create the environment and install the locked dependencies:

```bash
uv sync --dev
```

Copy the environment template for local configuration:

```bash
cp .env.example .env
```

The application can start without API keys during this foundation phase. Do not
commit `.env`; it is ignored by Git.

## Run

```bash
uv run agentic-chatbot
```

Expected output:

```text
INFO agentic_chatbot.main: Agentic AI chatbot foundation started successfully.
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
├── config.py
├── logging_config.py
└── main.py
tests/
├── test_config.py
└── test_main.py
```

## Configuration

Configuration is read from environment variables and, when present, a local
`.env` file. See `.env.example` for the supported variables. Sensitive values are
represented as secret values and are never written to startup logs.

Later phases will add the agent graph and integrations incrementally.
