"""Simple Streamlit UI that communicates exclusively with FastAPI."""

from typing import Any, Literal

import streamlit as st

from agentic_chatbot.api_client import AgentApiClient, BackendClientError
from agentic_chatbot.api_models import (
    ConversationResponse,
    ConversationStateResponse,
    HistoryMessageResponse,
)
from agentic_chatbot.config import load_settings
from agentic_chatbot.ui_state import (
    StreamViewState,
    choose_active_thread,
    extract_source_references,
)

CURRENT_THREAD_KEY = "current_thread_id"
UPLOAD_NONCE_KEY = "pdf_upload_nonce"
FLASH_KEY = "ui_flash"


def _show_backend_error(error: BackendClientError) -> None:
    st.error(str(error))


def _show_flash() -> None:
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.toast(flash)


def _render_sources(content: str) -> None:
    sources = extract_source_references(content)
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for source in sources:
            st.markdown(f"- `{source.filename}` — page {source.page}")


def _render_message(message: HistoryMessageResponse) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.role == "assistant":
            _render_sources(message.content)


def _create_conversation(client: AgentApiClient) -> None:
    conversation = client.create_conversation()
    st.session_state[CURRENT_THREAD_KEY] = str(conversation.thread_id)
    st.session_state[FLASH_KEY] = "New conversation created."


def _render_conversation_sidebar(
    client: AgentApiClient,
    conversations: list[ConversationResponse],
    current: ConversationResponse,
) -> None:
    with st.sidebar:
        st.header("Conversations")
        if st.button("＋ New Chat", width="stretch", type="primary"):
            try:
                _create_conversation(client)
            except BackendClientError as error:
                _show_backend_error(error)
            else:
                st.rerun()

        for conversation in conversations:
            thread_id = str(conversation.thread_id)
            selected = thread_id == str(current.thread_id)
            if st.button(
                conversation.title,
                key=f"select_{thread_id}",
                width="stretch",
                disabled=selected,
            ):
                st.session_state[CURRENT_THREAD_KEY] = thread_id
                st.rerun()

        st.caption(f"Thread ID: `{current.thread_id}`")

        with st.expander("Rename or delete"):
            with st.form(f"rename_{current.thread_id}"):
                new_title = st.text_input("Conversation title", value=current.title)
                rename_submitted = st.form_submit_button(
                    "Rename", width="stretch"
                )
            if rename_submitted:
                try:
                    client.rename_conversation(str(current.thread_id), new_title)
                except BackendClientError as error:
                    _show_backend_error(error)
                else:
                    st.session_state[FLASH_KEY] = "Conversation renamed."
                    st.rerun()

            confirm_delete = st.checkbox(
                "I understand this removes this conversation",
                key=f"confirm_delete_{current.thread_id}",
            )
            if st.button(
                "Delete conversation",
                key=f"delete_{current.thread_id}",
                disabled=not confirm_delete,
                width="stretch",
            ):
                try:
                    client.delete_conversation(str(current.thread_id))
                except BackendClientError as error:
                    _show_backend_error(error)
                else:
                    remaining = [
                        conversation
                        for conversation in conversations
                        if conversation.thread_id != current.thread_id
                    ]
                    st.session_state[CURRENT_THREAD_KEY] = choose_active_thread(
                        remaining, None
                    )
                    st.session_state[FLASH_KEY] = "Conversation deleted."
                    st.rerun()

        _render_document_sidebar(client, str(current.thread_id))


def _render_document_sidebar(client: AgentApiClient, thread_id: str) -> None:
    st.divider()
    st.subheader("Conversation PDFs")
    nonce = st.session_state.setdefault(UPLOAD_NONCE_KEY, 0)
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        max_upload_size=20,
        key=f"pdf_upload_{thread_id}_{nonce}",
        help="The backend extracts, embeds, and stores this PDF for this thread.",
    )
    if st.button(
        "Ingest PDF",
        key=f"ingest_{thread_id}_{nonce}",
        disabled=uploaded_pdf is None,
        width="stretch",
    ):
        assert uploaded_pdf is not None
        try:
            with st.spinner("Extracting and embedding PDF…"):
                document = client.upload_pdf(
                    thread_id,
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                )
        except BackendClientError as error:
            _show_backend_error(error)
        else:
            st.session_state[UPLOAD_NONCE_KEY] = nonce + 1
            st.session_state[FLASH_KEY] = (
                f"Ingested {document.source_filename}: "
                f"{document.page_count} pages, {document.chunk_count} chunks."
            )
            st.rerun()

    try:
        documents = client.list_documents(thread_id)
    except BackendClientError as error:
        _show_backend_error(error)
        return
    if not documents:
        st.caption("No PDFs uploaded to this conversation.")
        return
    for document in documents:
        st.markdown(f"**{document.source_filename}**")
        st.caption(
            f"{document.page_count} pages · {document.chunk_count} chunks"
        )


def _resolve_approval(
    client: AgentApiClient,
    thread_id: str,
    decision: Literal["approve", "reject"],
) -> None:
    try:
        with st.spinner(f"Submitting {decision} decision…"):
            client.respond_to_approval(thread_id, decision)
    except BackendClientError as error:
        _show_backend_error(error)
        return
    st.session_state[FLASH_KEY] = (
        "Paper trade approved." if decision == "approve" else "Paper trade rejected."
    )
    st.rerun()


def _render_pending_approval(
    client: AgentApiClient,
    thread_id: str,
    approval: dict[str, Any],
) -> None:
    st.warning("Approval required — PAPER TRADING ONLY")
    st.markdown(
        f"**Action:** `paper_buy_stock`  \n"
        f"**Ticker:** `{approval.get('ticker', 'unknown')}`  \n"
        f"**Quantity:** `{approval.get('quantity', 'unknown')}`"
    )
    approve_column, reject_column = st.columns(2)
    if approve_column.button(
        "Approve",
        key=f"approve_{thread_id}",
        type="primary",
        width="stretch",
    ):
        _resolve_approval(client, thread_id, "approve")
    if reject_column.button(
        "Reject",
        key=f"reject_{thread_id}",
        width="stretch",
    ):
        _resolve_approval(client, thread_id, "reject")


def _stream_chat_message(
    client: AgentApiClient,
    thread_id: str,
    prompt: str,
) -> StreamViewState:
    presentation = StreamViewState()
    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        answer_placeholder = st.empty()
        try:
            for event in client.stream_message(thread_id, prompt):
                presentation.apply(event)
                if presentation.tool_status:
                    tool_placeholder.info(presentation.tool_status)
                else:
                    tool_placeholder.empty()
                if presentation.assistant_text:
                    answer_placeholder.markdown(
                        f"{presentation.assistant_text}▌"
                    )
                if presentation.error:
                    st.error(presentation.error)
        except BackendClientError as error:
            presentation.error = str(error)
            st.error(presentation.error)

        tool_placeholder.empty()
        if presentation.assistant_text:
            answer_placeholder.markdown(presentation.assistant_text)
            _render_sources(presentation.assistant_text)
        elif presentation.pending_approval is not None:
            answer_placeholder.info("Waiting for your approval below.")
        elif not presentation.error:
            answer_placeholder.warning("The agent returned no visible response.")
    return presentation


def _load_initial_state(
    client: AgentApiClient,
) -> tuple[list[ConversationResponse], ConversationResponse, ConversationStateResponse]:
    client.health()
    conversations = client.list_conversations()
    current_thread_id = choose_active_thread(
        conversations,
        st.session_state.get(CURRENT_THREAD_KEY),
    )
    if current_thread_id is None:
        conversation = client.create_conversation()
        conversations = [conversation]
        current_thread_id = str(conversation.thread_id)
    st.session_state[CURRENT_THREAD_KEY] = current_thread_id
    current = next(
        conversation
        for conversation in conversations
        if str(conversation.thread_id) == current_thread_id
    )
    state = client.get_conversation_state(current_thread_id)
    return conversations, current, state


def render_app(client: AgentApiClient) -> None:
    """Render one Streamlit run using FastAPI as the source of truth."""

    _show_flash()
    try:
        conversations, current, state = _load_initial_state(client)
    except BackendClientError as error:
        st.error(str(error))
        st.info(
            "Start FastAPI with: `uv run uvicorn agentic_chatbot.api:app --reload`"
        )
        st.stop()

    _render_conversation_sidebar(client, conversations, current)
    st.title(current.title)
    st.caption("Gemini + LangGraph · persistent thread")

    for message in state.messages:
        _render_message(message)

    thread_id = str(current.thread_id)
    if state.pending_approval is not None:
        _render_pending_approval(
            client,
            thread_id,
            state.pending_approval.model_dump(),
        )

    prompt = st.chat_input(
        "Ask Gemini or request a tool…",
        disabled=state.pending_approval is not None,
    )
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    result = _stream_chat_message(client, thread_id, prompt)
    if result.completed or result.pending_approval is not None:
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Agentic AI Chatbot",
        page_icon="🤖",
        layout="wide",
    )
    settings = load_settings()
    with AgentApiClient(settings.backend_api_url) as client:
        render_app(client)


if __name__ == "__main__":
    main()
