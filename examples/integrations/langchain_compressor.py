"""RAGDefender inside LangChain, as a document compressor.

    pip install "ragdefender[langchain]" langchain-classic

Two ways to use it: wrap a retriever (filter runs on every retrieval) or call
``compress_documents`` on a list of Documents. The toy retriever below stands
in for ``vectorstore.as_retriever()``; it returns the same passages for any query.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragdefender.integrations import RAGDefenderCompressor

from _passages import PASSAGES, QUERY, report


class ToyRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


def main() -> None:
    docs = [Document(page_content=t, metadata={"pos": i}) for i, t in enumerate(PASSAGES)]
    compressor = RAGDefenderCompressor(task_type="single_hop", device="cpu")

    # 1. Direct call on retrieved Documents
    safe = compressor.compress_documents(docs, QUERY)
    report("compress_documents()", (d.metadata["pos"] for d in safe))

    # 2. Wrapped retriever (ContextualCompressionRetriever from langchain-classic / langchain)
    try:
        safe_retriever = compressor.wrap_retriever(ToyRetriever(docs=docs))
    except ImportError as exc:
        print(f"skipping wrap_retriever(): {exc}")
        return
    report("wrap_retriever().invoke()", (d.metadata["pos"] for d in safe_retriever.invoke(QUERY)))


if __name__ == "__main__":
    main()
