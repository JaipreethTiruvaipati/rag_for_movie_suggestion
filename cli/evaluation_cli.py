#!/usr/bin/env python3
import argparse
import json
import os

from hybrid_search_cli import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load golden dataset
    golden_path = "data/golden_dataset.json"
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {golden_path} not found.")
        return

    # Load movie documents
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("Error: data/movies.json not found.")
        return

    search = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        results = search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [res["title"] for res in results]

        # Precision@k: fraction of retrieved docs that are relevant
        relevant_set = set(relevant_docs)
        hits = sum(1 for title in retrieved_titles if title in relevant_set)
        precision = hits / limit if limit > 0 else 0.0

        # Recall@k: fraction of relevant docs that were retrieved
        recall = hits / len(relevant_docs) if relevant_docs else 0.0

        # F1 Score: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_docs)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}")
        print()


if __name__ == "__main__":
    main()
