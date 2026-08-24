# Agentic AI Chatbot

An educational project for learning how modern agentic systems are assembled with
LangGraph, LangChain, Google Gemini, LangSmith, SQLite, and local retrieval.

Phase 6 adds conversation-scoped PDF retrieval. PDFs are extracted, split into
chunks, embedded with Gemini, and stored in local persistent Chroma. Gemini can
call `search_documents` when a question needs uploaded material; PDF content is
not automatically added to every prompt.

This project never connects to a brokerage, places real orders, or uses real
money.

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

Omitting `--thread-id` always creates a fresh conversation UUID. Its initial
title is generated from the first user message by collapsing whitespace and
truncating it to 60 characters; this does not require an extra model call.

Manage saved conversations without starting Gemini:

```bash
uv run agentic-chatbot --list-conversations
uv run agentic-chatbot --rename 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e "LangGraph lesson"
uv run agentic-chatbot --delete 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
```

`--list-conversations` prints the UUID, title, and last-updated timestamp.
Deleting removes the metadata record and asks the LangGraph checkpointer to
delete that thread through its public API.

Ingest and inspect PDFs associated with a conversation:

```bash
THREAD_ID="paste-the-conversation-uuid"
uv run agentic-chatbot --thread-id "$THREAD_ID" --ingest-pdf ./documents/guide.pdf
uv run agentic-chatbot --thread-id "$THREAD_ID" --list-documents
uv run agentic-chatbot --thread-id "$THREAD_ID"
```

PDF ingestion uses the Gemini API to generate embeddings. `PyPDFLoader` extracts
embedded text; image-only scanned PDFs require OCR, which is not part of this
phase.

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
├── conversations.py
├── graph.py
├── logging_config.py
├── model.py
├── persistence.py
├── rag.py
├── tools/
│   ├── __init__.py
│   ├── calculator.py
│   ├── document_search.py
│   ├── paper_trading.py
│   ├── weather.py
│   └── web_search.py
└── main.py
data/
├── chroma/                       # PDF vectors; created locally; ignored
├── conversations.sqlite          # app metadata; created locally; ignored
└── langgraph_checkpoints.sqlite  # graph state; created locally; ignored
tests/
├── test_calculator.py
├── test_cli.py
├── test_config.py
├── test_conversations.py
├── test_graph.py
├── test_main.py
├── test_model.py
├── test_paper_trading.py
├── test_persistence.py
├── test_rag.py
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

The agent/model node binds all five tool schemas to Gemini. After Gemini replies,
`tools_condition` checks whether its message contains tool calls. Tool calls go to
`ToolNode`; ordinary answers go to `END`. Tool results loop back to Gemini so it
can turn raw data into a natural-language answer.

The graph shape remains explicit and unchanged in Phase 5. The
`paper_buy_stock` tool calls `interrupt()` before its simulated execution. That
pauses `ToolNode` and stores a structured approval request containing the ticker
and quantity. Harmless tools contain no interrupt and continue immediately.

At compilation, the graph receives a SQLite `SqliteSaver` checkpointer. Every
streamed graph execution also receives this configuration:

```python
{"configurable": {"thread_id": conversation_uuid}}
```

Before a run, the checkpointer loads the latest state for that thread. After graph
steps, it saves new state snapshots. The CLI therefore submits only the newest
human message; LangGraph restores and combines the earlier messages itself.

## Two SQLite databases

`data/langgraph_checkpoints.sqlite` is owned by LangGraph. It contains graph
state such as accumulated messages, pending interrupts, and checkpoint
bookkeeping needed to resume execution. Application code does not inspect or
edit its internal tables.

`data/conversations.sqlite` is owned by this application. Its `conversations`
table contains only `thread_id`, `title`, `created_at`, and `updated_at`. Those
fields support menus and lifecycle actions without treating internal checkpoint
rows as application data.

The same `thread_id` appears in both systems. The metadata record says how to
present a conversation, while the LangGraph checkpoint associated with that ID
restores what was said. When deleting, the application calls the checkpointer's
documented `delete_thread()` method first, then deletes its own metadata; it
never manually manipulates undocumented checkpoint tables.

## PDF ingestion and RAG

```text
PDF
 │
 ▼
PyPDFLoader extracts one document per page
 │
 ▼
RecursiveCharacterTextSplitter creates overlapping chunks
 │
 ▼
Gemini generates an embedding vector for each chunk
 │
 ▼
Local Chroma stores vectors, text, and metadata
```

Every chunk contains `thread_id`, `document_id`, `source_filename`,
`page_number`, and `chunk_index`. Retrieval always applies a `thread_id` filter
inside the tool. The thread is captured when the tool is constructed and is not
an argument Gemini can change.

When Gemini calls `search_documents`, the query is embedded and Chroma returns
the most similar chunks from that conversation only. Results include citation
information such as `guide.pdf, page 3`. Gemini then uses those passages to
compose the answer and cite its sources.

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
- `paper_buy_stock`: proposes a simulated purchase with a ticker and quantity,
  then pauses for explicit approval. Approval returns a paper-only result;
  rejection returns a tool result saying nothing was executed. It has no
  brokerage integration and cannot move money.
- `search_documents`: searches only the PDFs attached to the active conversation
  and returns relevant passages with filename, page number, and document ID.
  Gemini decides when document retrieval is needed.

## Human approval flow

```text
Gemini requests paper_buy_stock
              │
              ▼
       interrupt(payload)
              │
       checkpoint is saved
              │
       approve or reject
              │
              ▼
     Command(resume=decision)
              │
              ▼
 Tool result → Gemini → streamed answer
```

At an approval prompt, entering `exit` or `quit` closes the program without
resolving the request. Restart with the same `--thread-id` and the persisted
approval prompt is displayed again.

## Configuration

Configuration is read from environment variables and, when present, a local
`.env` file. See `.env.example` for all supported values. Sensitive values are
represented as secret values and are never written to startup logs.

`GEMINI_MODEL` selects the model and defaults to `gemini-2.5-flash`.
`GEMINI_EMBEDDING_MODEL` selects the PDF embedding model and defaults to
`gemini-embedding-001`.
`LOG_LEVEL` controls this project's logs; routine HTTP-client request logs are
suppressed so they do not interrupt streamed assistant output.
`CHECKPOINT_DB_PATH` selects the local checkpoint file and defaults to
`data/langgraph_checkpoints.sqlite`.
`CONVERSATION_DB_PATH` selects the separate application metadata file and
defaults to `data/conversations.sqlite`.
`CHROMA_DB_PATH` selects the local vector store directory and defaults to
`data/chroma`.

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

## Try conversation management, persistence, and tools

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
You: Paper buy 5 shares of AAPL.
Approval required — PAPER TRADING ONLY
Action: paper_buy_stock
Ticker: AAPL
Quantity: 5
Approve or reject? [approve/reject]: approve
Assistant: [A streamed explanation that the simulated paper trade executed.]
You: According to the uploaded guide, how are checkpoints restored?
Assistant: [Gemini calls search_documents and answers with guide.pdf, page N.]
You: quit
Goodbye!
```

Reusing the displayed UUID restores its saved graph state. A new UUID starts with
no messages from other conversations.

List and manage the record after the first session:

```bash
uv run agentic-chatbot --list-conversations
uv run agentic-chatbot --rename 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e "Favorite color"
uv run agentic-chatbot --delete 7d91f4eb-276f-4a24-bb27-c5ee0b1b8f9e
```

Real brokerage integration and real monetary transactions are intentionally not
part of this project. OCR for scanned PDFs, graphical UI, and FastAPI remain
future phases.
