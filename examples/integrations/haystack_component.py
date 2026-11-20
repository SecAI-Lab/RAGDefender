"""RAGDefender inside a Haystack 2.x pipeline, between retriever and prompt builder.

    pip install "ragdefender[haystack]"

The toy retriever component stands in for ``InMemoryBM25Retriever`` & co.
"""
from typing import List

from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder

from ragdefender.integrations import RAGDefenderFilter

from _passages import PASSAGES, QUERY, report

TEMPLATE = """Answer using only the passages below.
{% for doc in documents %}
[{{ doc.meta.pos }}] {{ doc.content }}
{% endfor %}
Question: {{ question }}"""


@component
class ToyRetriever:
    def __init__(self, docs: List[Document]) -> None:
        self.docs = docs

    @component.output_types(documents=List[Document])
    def run(self, query: str):
        return {"documents": self.docs}


def main() -> None:
    docs = [Document(content=t, meta={"pos": i}) for i, t in enumerate(PASSAGES)]

    pipe = Pipeline()
    pipe.add_component("retriever", ToyRetriever(docs))
    pipe.add_component("defender", RAGDefenderFilter(task_type="single_hop", device="cpu"))
    pipe.add_component(
        "prompt", PromptBuilder(template=TEMPLATE, required_variables=["documents", "question"])
    )
    pipe.connect("retriever.documents", "defender.documents")
    pipe.connect("defender.documents", "prompt.documents")

    result = pipe.run(
        {"retriever": {"query": QUERY}, "defender": {"query": QUERY}, "prompt": {"question": QUERY}},
        include_outputs_from={"defender"},
    )
    report("Pipeline retriever → RAGDefenderFilter → PromptBuilder",
           (d.meta["pos"] for d in result["defender"]["documents"]))
    print("\nPrompt handed to the generator:\n" + result["prompt"]["prompt"])


if __name__ == "__main__":
    main()
