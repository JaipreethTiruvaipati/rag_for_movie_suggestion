#!/usr/bin/env python3
import argparse
import json
import os

from dotenv import load_dotenv
from google import genai


def load_movies():
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            return json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return None


def get_gemini_client():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return None

    return genai.Client(api_key=api_key)


def get_full_result_documents(results, documents):
    document_map = {doc["id"]: doc for doc in documents}
    full_results = []

    for result in results:
        full_doc = document_map.get(result["id"], {})
        full_results.append(
            {
                **result,
                "description": full_doc.get("description", result.get("description", "")),
            }
        )

    return full_results


def run_rag(query: str) -> None:
    from hybrid_search_cli import HybridSearch

    documents = load_movies()
    if documents is None:
        return

    search = HybridSearch(documents)
    results = search.rrf_search(query=query, k=60, limit=5)
    full_results = get_full_result_documents(results, documents)

    client = get_gemini_client()
    if client is None:
        return

    docs = []
    for i, result in enumerate(full_results, 1):
        docs.append(
            (
                f"{i}. Title: {result.get('title', '')}\n"
                f"Description: {result.get('description', '')}\n"
                f"RRF Score: {result.get('rrf_score', 0.0):.4f}"
            )
        )
    docs = "\n\n".join(docs)

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )

    print("Search Results:")
    for result in results:
        print(f"- {result.get('title', '')}")

    print("\nRAG Response:")
    print(response.text.strip() if response.text else "")


def run_summarize(query: str, limit: int) -> None:
    from hybrid_search_cli import HybridSearch

    documents = load_movies()
    if documents is None:
        return

    search = HybridSearch(documents)
    results = search.rrf_search(query=query, k=60, limit=limit)
    full_results = get_full_result_documents(results, documents)

    client = get_gemini_client()
    if client is None:
        return

    result_blocks = []
    for i, result in enumerate(full_results, 1):
        result_blocks.append(
            (
                f"{i}. Title: {result.get('title', '')}\n"
                f"Description: {result.get('description', '')}\n"
                f"RRF Score: {result.get('rrf_score', 0.0):.4f}"
            )
        )
    formatted_results = "\n\n".join(result_blocks)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{formatted_results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )

    print("Search Results:")
    for result in results:
        print(f"  - {result.get('title', '')}")

    print("\nLLM Summary:")
    print(response.text.strip() if response.text else "")


def run_citations(query: str, limit: int) -> None:
    from hybrid_search_cli import HybridSearch

    documents = load_movies()
    if documents is None:
        return

    search = HybridSearch(documents)
    results = search.rrf_search(query=query, k=60, limit=limit)
    full_results = get_full_result_documents(results, documents)

    client = get_gemini_client()
    if client is None:
        return

    document_blocks = []
    for i, result in enumerate(full_results, 1):
        document_blocks.append(
            (
                f"[{i}] Title: {result.get('title', '')}\n"
                f"Description: {result.get('description', '')}\n"
                f"RRF Score: {result.get('rrf_score', 0.0):.4f}"
            )
        )
    formatted_documents = "\n\n".join(document_blocks)

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{formatted_documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )

    print("Search Results:")
    for result in results:
        print(f"  - {result.get('title', '')}")

    print("\nLLM Answer:")
    print(response.text.strip() if response.text else "")


def run_question(question: str, limit: int) -> None:
    from hybrid_search_cli import HybridSearch

    documents = load_movies()
    if documents is None:
        return

    search = HybridSearch(documents)
    results = search.rrf_search(query=question, k=60, limit=limit)
    full_results = get_full_result_documents(results, documents)

    client = get_gemini_client()
    if client is None:
        return

    context_blocks = []
    for i, result in enumerate(full_results, 1):
        context_blocks.append(
            (
                f"{i}. Title: {result.get('title', '')}\n"
                f"Description: {result.get('description', '')}\n"
                f"RRF Score: {result.get('rrf_score', 0.0):.4f}"
            )
        )
    context = "\n\n".join(context_blocks)

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )

    print("Search Results:")
    for result in results:
        print(f"  - {result.get('title', '')}")

    print("\nAnswer:")
    print(response.text.strip() if response.text else "")


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform multi-document summarization over search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with source citations from search results"
    )
    citations_parser.add_argument("query", type=str, help="Search query to answer")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a user question conversationally from search results"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            run_rag(query)
        case "summarize":
            run_summarize(args.query, args.limit)
        case "citations":
            run_citations(args.query, args.limit)
        case "question":
            run_question(args.question, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
