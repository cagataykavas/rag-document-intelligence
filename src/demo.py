from .retrieval import Document, SparseRetriever, chunk_documents, reciprocal_rank


def main() -> None:
    documents = [
        Document("rl", "Reinforcement learning optimizes sequential decisions using rewards, policies and value functions.", "rl_notes.md"),
        Document("rag", "Retrieval augmented generation retrieves relevant evidence before a language model generates an answer. Evaluation should measure retrieval separately from generation.", "rag_notes.md"),
        Document("xai", "Model interpretability methods inspect predictions, representations, attribution and causal interventions.", "xai_notes.md"),
    ]
    retriever = SparseRetriever().fit(chunk_documents(documents, words_per_chunk=30, overlap=5))
    queries = [("How does RAG obtain evidence?", "rag"), ("What optimizes sequential decisions?", "rl"), ("How can model decisions be inspected?", "xai")]
    rr = []
    for query, relevant in queries:
        results = retriever.search(query, k=3)
        rr.append(reciprocal_rank(results, relevant))
        print(f"\nQUERY: {query}")
        for result in results:
            print(f"  {result['score']:.3f} [{result['source']}] {result['text']}")
    print(f"\nMRR: {sum(rr) / len(rr):.3f}")


if __name__ == "__main__":
    main()
