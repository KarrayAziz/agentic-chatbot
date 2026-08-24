# Agentic AI Chatbot

An educational project for learning how modern agentic systems are assembled with
LangGraph, LangChain, Google Gemini, LangSmith, SQLite, and local retrieval.

Phase 3 persists LangGraph conversation state in a local SQLite checkpoint
database. Every conversation has a UUID thread ID, so it can be resumed after the
Python process exits while preserving streaming and the existing tool-calling
loop.

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

Add your Google AI Studio and Tavily API keys to `.env`:

```dotenv
GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
```

Do not commit `.env`; it is ignored by Git.

## Run

Start a new conversation:

```bash
uv run agentic-chatbot
```

The application creates and displays a UUID:

```text
Gemini chatbot ready. Type 'exit' or 'quit' to stop.
Conversation ID: 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
You:
```

Save that ID. To continue the same conversation after restarting:

```bash
uv run agentic-chatbot --thread-id 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
```

Omitting `--thread-id` always creates a fresh conversation UUID.

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
├── persistence.py
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── weather.py
│   └── web_search.py
└── main.py
data/
└── langgraph_checkpoints.sqlite  # created locally; ignored by Git
tests/
├── test_calculator.py
├── test_cli.py
├── test_config.py
├── test_graph.py
├── test_main.py
├── test_model.py
├── test_persistence.py
├── test_weather.py
└── test_web_search.py
```

## Graph architecture

```text
                     no tool call
                  ┌──────────────→ END
                  │
START ──→ agent/model node
                  │
                  │ tool call requested
                  ▼
              tools node
                  │
                  └──────────────→ agent/model node
                                      (interprets tool result)
```

The graph is assembled directly with `StateGraph`, `MessagesState`, `START`,
`END`, conditional edges, `ToolNode`, and `tools_condition`. No prebuilt agent
abstraction is used.

The agent/model node binds all three tool schemas to Gemini. After Gemini replies,
`tools_condition` checks whether its message contains tool calls. Tool calls go to
`ToolNode`; ordinary answers go to `END`. Tool results loop back to Gemini so it
can turn raw data into a natural-language answer.

At compilation, the graph receives a SQLite `SqliteSaver` checkpointer. Every
streamed graph execution also receives this configuration:

```python
{"configurable": {"thread_id": conversation_uuid}}
```

Before a run, the checkpointer loads the latest state for that thread. After graph
steps, it saves new state snapshots. The CLI therefore submits only the newest
human message; LangGraph restores and combines the earlier messages itself.

The checkpoint database is conceptually separate from application conversation
metadata. It contains LangGraph execution state only. There is intentionally no
custom table for conversation names, listing, renaming, or deletion yet.

The graph structure is unchanged for streaming. The CLI now executes it with:

```python
graph.stream(
    {"messages": messages},
    stream_mode=["messages", "values"],
    version="v2",
)
```

- `messages` events contain `(message_chunk, metadata)` pairs. Text chunks from
  the `agent` node are written immediately to the terminal.
- `values` events contain full state snapshots. The last snapshot becomes the
  conversation history for the next user turn.

Metadata filtering ensures only output from the agent/model node is displayed.
Structured tool-call blocks and tool results remain internal. The completed AI
message already exists in the final state, but is not printed again.

## Streaming behavior

For a normal response, text begins appearing as Gemini generates it. For a tool
request, Gemini first emits a structured tool call, so there may initially be no
visible text. The graph runs the tool, loops back to Gemini, and then streams the
final natural-language answer. A short pause during weather or web search is
normal because the external service must finish before Gemini can use its result.

## Available tools

- `calculator`: parses an allowlist of arithmetic syntax with Python's AST. It
  never uses `eval()` and rejects names, function calls, code, excessive powers,
  non-finite values, and overly large results.
- `get_current_weather`: resolves a location with Open-Meteo's geocoding API and
  returns current temperature, apparent temperature, humidity, condition, and
  wind speed from its forecast API. No weather API key is required.
- `tavily_search`: returns up to five current web-search results. It requires
  `TAVILY_API_KEY`.

## Configuration

Configuration is read from environment variables and, when present, a local
`.env` file. See `.env.example` for all supported values. Sensitive values are
represented as secret values and are never written to startup logs.

`GEMINI_MODEL` selects the model and defaults to `gemini-2.5-flash`.
`LOG_LEVEL` controls this project's logs; routine HTTP-client request logs are
suppressed so they do not interrupt streamed assistant output.
`CHECKPOINT_DB_PATH` selects the local checkpoint file and defaults to
`data/langgraph_checkpoints.sqlite`.

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

## Try persistence and tools

```text
Command: uv run agentic-chatbot
Conversation ID: 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
You: My favorite color is blue.
Assistant: I will remember that.
You: quit
Goodbye!

Command: uv run agentic-chatbot --thread-id 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
Conversation ID: 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
You: What is my favorite color?
Assistant: Your favorite color is blue.

You: What is 837 * 92?
Assistant: 837 × 92 is 77,004.  [appears progressively]
You: What is the weather in Tunis?
Assistant: [A progressively displayed answer after the weather lookup.]
You: Search the web for the latest LangGraph release.
Assistant: [A progressively displayed answer after Tavily search.]
You: Hello, how are you?
Assistant: [A progressively displayed response without needing a tool.]
You: quit
Goodbye!
```

Reusing the displayed UUID restores its saved graph state. A new UUID starts with
no messages from other conversations.

Conversation metadata management, human approval, retrieval, and UI layers remain
future phases.
