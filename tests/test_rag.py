"""Tests for PDF chunking, persistent retrieval, and thread isolation."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_chatbot.rag import CHROMA_COLLECTION_NAME, DocumentRAGService
from agentic_chatbot.tools.document_search import create_document_search_tool


class KeywordEmbeddings(Embeddings):
    """Small deterministic embeddings used instead of a paid API in tests."""

    WORDS = ("oranges", "rockets", "tunis", "langgraph")

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(word)) for word in cls.WORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class TwoPageLoader(BaseLoader):
    """Loader fixture with the same page metadata shape as PyPDFLoader."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self):
        yield Document(
            page_content=("Oranges are grown in sunny orchards. " * 4),
            metadata={"source": self.file_path, "page": 0},
        )
        yield Document(
            page_content=("Rockets use engines to reach space. " * 4),
            metadata={"source": self.file_path, "page": 1},
        )


class ThreadTextLoader(BaseLoader):
    """Use the filename to create distinguishable per-thread content."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self):
        yield Document(
            page_content=f"Oranges information unique to {Path(self.file_path).stem}.",
            metadata={"page": 0},
        )


def _vector_store(path: Path) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=KeywordEmbeddings(),
        persist_directory=str(path),
    )


def test_ingestion_preserves_chunk_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "orchards.pdf"
    pdf_path.write_bytes(b"%PDF-test-fixture")
    service = DocumentRAGService(
        _vector_store(tmp_path / "chroma"),
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=80,
            chunk_overlap=10,
            add_start_index=True,
        ),
        loader_factory=TwoPageLoader,
    )

    summary = service.ingest_pdf(
        pdf_path,
        "thread-a",
        document_id="document-a",
    )
    stored = service.vector_store.get(
        where={"document_id": "document-a"},
        include=["metadatas"],
    )

    assert summary.source_filename == "orchards.pdf"
    assert summary.page_count == 2
    assert summary.chunk_count > 2
    assert all(
        metadata["source_filename"] == "orchards.pdf"
        and metadata["document_id"] == "document-a"
        and metadata["thread_id"] == "thread-a"
        and metadata["page_number"] in {1, 2}
        and isinstance(metadata["chunk_index"], int)
        for metadata in stored["metadatas"]
    )


def test_retrieval_persists_and_returns_source_information(tmp_path: Path) -> None:
    pdf_path = tmp_path / "handbook.pdf"
    pdf_path.write_bytes(b"%PDF-test-fixture")
    chroma_path = tmp_path / "chroma"
    service = DocumentRAGService(
        _vector_store(chroma_path),
        loader_factory=TwoPageLoader,
    )
    service.ingest_pdf(pdf_path, "thread-a", document_id="handbook-id")

    reopened_service = DocumentRAGService(_vector_store(chroma_path))
    tool = create_document_search_tool(reopened_service, "thread-a")
    assert "thread_id" not in tool.args
    result = tool.invoke({"query": "How do rockets reach space?", "max_results": 1})

    assert result["status"] == "ok"
    assert result["result_count"] == 1
    assert "Rockets" in result["results"][0]["content"]
    assert result["results"][0]["document_id"] == "handbook-id"
    assert result["results"][0]["citation"] == "handbook.pdf, page 2"


def test_search_and_document_listing_are_isolated_by_thread(tmp_path: Path) -> None:
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.write_bytes(b"%PDF-test-fixture")
    second_pdf.write_bytes(b"%PDF-test-fixture")
    service = DocumentRAGService(
        _vector_store(tmp_path / "chroma"),
        loader_factory=ThreadTextLoader,
    )
    service.ingest_pdf(first_pdf, "thread-a", document_id="first-id")
    service.ingest_pdf(second_pdf, "thread-b", document_id="second-id")

    first_results = service.search("oranges", "thread-a")
    first_documents = service.list_documents("thread-a")

    assert {document.metadata["thread_id"] for document in first_results} == {
        "thread-a"
    }
    assert "first" in first_results[0].page_content
    assert [document.document_id for document in first_documents] == ["first-id"]
