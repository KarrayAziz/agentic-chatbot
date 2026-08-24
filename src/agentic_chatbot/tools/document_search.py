"""Thread-scoped document retrieval tool construction."""

from typing import Any

from langchain_core.tools import BaseTool, tool

from agentic_chatbot.rag import DocumentRAGService


def create_document_search_tool(
    rag_service: DocumentRAGService, thread_id: str
) -> BaseTool:
    """Create a retrieval tool permanently scoped to one conversation."""

    @tool("search_documents")
    def search_documents(query: str, max_results: int = 4) -> dict[str, Any]:
        """Search PDFs uploaded to this conversation for relevant passages.

        Use this for questions about uploaded PDFs or when the user asks you to
        consult their documents. Cite each used passage with its returned
        source filename and page number. Do not invent document sources.
        """

        documents = rag_service.search(query, thread_id, k=max_results)
        results = [
            {
                "content": document.page_content,
                "source_filename": document.metadata["source_filename"],
                "page_number": document.metadata["page_number"],
                "document_id": document.metadata["document_id"],
                "citation": (
                    f"{document.metadata['source_filename']}, "
                    f"page {document.metadata['page_number']}"
                ),
            }
            for document in documents
        ]
        return {
            "status": "ok" if results else "no_matches",
            "result_count": len(results),
            "results": results,
        }

    return search_documents
