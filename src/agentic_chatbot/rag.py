"""PDF ingestion and thread-scoped retrieval backed by local Chroma."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_chatbot.config import PROJECT_ROOT, Settings

CHROMA_COLLECTION_NAME = "agentic_chatbot_documents"
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """Summary of one PDF stored for one conversation thread."""

    document_id: str
    source_filename: str
    chunk_count: int
    page_count: int


@dataclass(slots=True)
class _DocumentAccumulator:
    source_filename: str
    chunk_count: int
    pages: set[int]


LoaderFactory = Callable[[str], BaseLoader]


class DocumentRAGService:
    """Load PDFs, create chunks, and retrieve only within one thread."""

    def __init__(
        self,
        vector_store: Chroma,
        *,
        splitter: RecursiveCharacterTextSplitter | None = None,
        loader_factory: LoaderFactory = PyPDFLoader,
    ) -> None:
        self.vector_store = vector_store
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            add_start_index=True,
        )
        self.loader_factory = loader_factory

    def ingest_pdf(
        self,
        pdf_path: Path,
        thread_id: str,
        *,
        document_id: str | None = None,
    ) -> IngestedDocument:
        """Extract, chunk, embed, and persist one PDF for a thread."""

        path = pdf_path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"PDF file does not exist: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files can be ingested.")
        if not thread_id.strip():
            raise ValueError("A conversation thread ID is required.")

        pages = self.loader_factory(str(path)).load()
        stored_document_id = document_id or str(uuid4())
        prepared_pages: list[Document] = []

        for page_position, page in enumerate(pages, start=1):
            if not page.page_content.strip():
                continue
            loader_page = page.metadata.get("page")
            page_number = (
                loader_page + 1 if isinstance(loader_page, int) else page_position
            )
            prepared_pages.append(
                Document(
                    page_content=page.page_content,
                    metadata={
                        "thread_id": thread_id,
                        "document_id": stored_document_id,
                        "source_filename": path.name,
                        "page_number": page_number,
                    },
                )
            )

        if not prepared_pages:
            raise ValueError("The PDF did not contain extractable text.")

        chunks = self.splitter.split_documents(prepared_pages)
        for chunk_index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = chunk_index

        vector_ids = [
            f"{stored_document_id}:{chunk_index}"
            for chunk_index in range(len(chunks))
        ]
        self.vector_store.add_documents(documents=chunks, ids=vector_ids)

        return IngestedDocument(
            document_id=stored_document_id,
            source_filename=path.name,
            chunk_count=len(chunks),
            page_count=len(prepared_pages),
        )

    def search(self, query: str, thread_id: str, *, k: int = 4) -> list[Document]:
        """Return semantically similar chunks restricted to one thread."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Document search query cannot be empty.")
        if not 1 <= k <= 8:
            raise ValueError("Document search result count must be between 1 and 8.")
        return self.vector_store.similarity_search(
            normalized_query,
            k=k,
            filter={"thread_id": thread_id},
        )

    def list_documents(self, thread_id: str) -> list[IngestedDocument]:
        """List distinct PDFs stored for a conversation thread."""

        stored = self.vector_store.get(
            where={"thread_id": thread_id},
            include=["metadatas"],
        )
        grouped: dict[str, _DocumentAccumulator] = {}
        for metadata in stored.get("metadatas") or []:
            document_id = str(metadata["document_id"])
            group = grouped.setdefault(
                document_id,
                _DocumentAccumulator(
                    source_filename=str(metadata["source_filename"]),
                    chunk_count=0,
                    pages=set(),
                ),
            )
            group.chunk_count += 1
            group.pages.add(int(metadata["page_number"]))

        return sorted(
            (
                IngestedDocument(
                    document_id=document_id,
                    source_filename=values.source_filename,
                    chunk_count=values.chunk_count,
                    page_count=len(values.pages),
                )
                for document_id, values in grouped.items()
            ),
            key=lambda document: (document.source_filename, document.document_id),
        )

    def delete_thread_documents(self, thread_id: str) -> int:
        """Delete all vectors owned by a conversation using Chroma's public API."""

        stored = self.vector_store.get(
            where={"thread_id": thread_id},
            include=[],
        )
        vector_ids = stored.get("ids") or []
        if vector_ids:
            self.vector_store.delete(ids=vector_ids)
        return len(vector_ids)


def resolve_chroma_path(path: Path) -> Path:
    """Resolve configured relative Chroma paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def create_document_rag_service(settings: Settings) -> DocumentRAGService:
    """Create the persistent Chroma service using Gemini embeddings."""

    if (
        settings.google_api_key is None
        or not settings.google_api_key.get_secret_value().strip()
    ):
        raise ValueError("GOOGLE_API_KEY is required for document embeddings.")

    persist_directory = resolve_chroma_path(settings.chroma_db_path)
    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key.get_secret_value(),
    )
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return DocumentRAGService(vector_store)
