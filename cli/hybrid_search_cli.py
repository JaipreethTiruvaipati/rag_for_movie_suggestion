#!/usr/bin/env python3
import argparse
import os

from keyword_search_cli import InvertedIndex
from semantic_search_cli import ChunkedSemanticSearch

def normalize(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    normalized = []
    for score in scores:
        if max_score == min_score:
            normalized.append(1.0)
        else:
            normalized.append((score - min_score) / (max_score - min_score))
    return normalized

def rrf_score(rank, k=60):
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [res["score"] for res in semantic_results]
        
        norm_bm25_scores = normalize(bm25_scores) if bm25_scores else []
        norm_semantic_scores = normalize(semantic_scores) if semantic_scores else []
        
        doc_map = {}
        for i, (doc, score) in enumerate(bm25_results):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {"bm25_score": 0.0, "semantic_score": 0.0}
            doc_map[doc_id]["bm25_score"] = norm_bm25_scores[i]
            
        for i, res in enumerate(semantic_results):
            doc_id = res["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {"bm25_score": 0.0, "semantic_score": 0.0}
            doc_map[doc_id]["semantic_score"] = norm_semantic_scores[i]
            
        results = []
        for doc_id, data in doc_map.items():
            bm25 = data["bm25_score"]
            semantic = data["semantic_score"]
            hybrid = (alpha * bm25) + ((1.0 - alpha) * semantic)
            
            orig_doc = next((d for d in self.documents if d["id"] == doc_id), {})
            title = orig_doc.get("title", "")
            desc = orig_doc.get("description", "")
            if len(desc) > 100:
                desc = desc[:100] + "..."
                
            results.append({
                "id": doc_id,
                "title": title,
                "description": desc,
                "hybrid_score": hybrid,
                "bm25_score": bm25,
                "semantic_score": semantic
            })
            
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:limit]

    def rrf_search(self, query, k, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        doc_map = {}
        for rank, (doc, _) in enumerate(bm25_results, 1):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0
                }
            doc_map[doc_id]["bm25_rank"] = rank
            doc_map[doc_id]["rrf_score"] += rrf_score(rank, k)
            
        for rank, res in enumerate(semantic_results, 1):
            doc_id = res["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0
                }
            doc_map[doc_id]["semantic_rank"] = rank
            doc_map[doc_id]["rrf_score"] += rrf_score(rank, k)
            
        results = []
        for doc_id, data in doc_map.items():
            orig_doc = next((d for d in self.documents if d["id"] == doc_id), {})
            title = orig_doc.get("title", "")
            desc = orig_doc.get("description", "")
            if len(desc) > 100:
                desc = desc[:100] + "..."
                
            results.append({
                "id": doc_id,
                "title": title,
                "description": desc,
                "rrf_score": data["rrf_score"],
                "bm25_rank": data["bm25_rank"],
                "semantic_rank": data["semantic_rank"]
            })
            
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:limit]

def run_normalize(scores):
    normalized = normalize(scores)
    for score in normalized:
        print(f"* {score:.4f}")

def run_weighted_search(query, alpha, limit):
    import json
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    search = HybridSearch(documents)
    results = search.weighted_search(query, alpha, limit)
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        print(f"  Hybrid Score: {res['hybrid_score']:.3f}")
        print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
        print(f"  {res['description']}")

def run_rrf_search(query, k, limit, enhance=None, rerank_method=None, evaluate=False):
    if enhance in ("spell", "rewrite", "expand") or rerank_method in ("individual", "batch"):
        import os
        import time
        from dotenv import load_dotenv
        from google import genai
        
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set")
            return
            
        client = genai.Client(api_key=api_key)
        
        prompt = None
        if enhance == "spell":
            prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        elif enhance == "rewrite":
            prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        elif enhance == "expand":
            prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
            
        if prompt:
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt,
            )
            
            if enhance == "expand":
                expanded_terms = response.text.strip()
                enhanced_query = f"{query} {expanded_terms}" if expanded_terms else query
            else:
                enhanced_query = response.text.strip()
                
            print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
            query = enhanced_query

    import json
    try:
        with open("data/movies.json", "r", encoding="utf-8") as f:
            documents = json.load(f)["movies"]
    except FileNotFoundError:
        print("data/movies.json not found! Please check your file paths.")
        return
        
    search = HybridSearch(documents)
    search_limit = limit * 5 if rerank_method in ("individual", "batch", "cross_encoder") else limit
    results = search.rrf_search(query, k, search_limit)

    if rerank_method == "individual":
        print(f"Re-ranking top {len(results)} results using individual method...")
        for res in results:
            prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {res.get("title", "")} - {res.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt,
            )
            try:
                res["rerank_score"] = float(response.text.strip())
            except ValueError:
                res["rerank_score"] = 0.0
            time.sleep(3)
            
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        results = results[:limit]
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
        
    elif rerank_method == "batch":
        print(f"Re-ranking top {len(results)} results using batch method...")
        doc_list_strs = []
        for res in results:
            doc_list_strs.append(f"ID: {res['id']}\nTitle: {res.get('title', '')}\nDescription: {res.get('description', '')[:100]}...")
        doc_list_str = "\n\n".join(doc_list_strs)
        
        prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )
        try:
            import json
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            ranked_ids = json.loads(clean_text.strip())
        except Exception:
            ranked_ids = []
            
        rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, 1)}
        for res in results:
            res["rerank_rank"] = rank_map.get(res["id"], float("inf"))
            
        results.sort(key=lambda x: x["rerank_rank"])
        results = results[:limit]
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
        
    elif rerank_method == "cross_encoder":
        print(f"Re-ranking top {len(results)} results using cross_encoder method...")
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        pairs = []
        for res in results:
            pairs.append([query, f"{res.get('title', '')} - {res.get('description', '')}"])
            
        scores = cross_encoder.predict(pairs)
        for idx, res in enumerate(results):
            res["cross_encoder_score"] = float(scores[idx])
            
        results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        results = results[:limit]
        print(f"Reciprocal Rank Fusion Results for '{query}' (k={k}):\n")
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']}")
        if rerank_method == "individual":
            print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
            print(f"   RRF Score: {res['rrf_score']:.3f}")
            print(f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
            print(f"   {res['description']}\n")
        elif rerank_method == "batch":
            print(f"   Re-rank Rank: {res.get('rerank_rank', 'N/A')}")
            print(f"   RRF Score: {res['rrf_score']:.3f}")
            print(f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
            print(f"   {res['description']}\n")
        elif rerank_method == "cross_encoder":
            print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")
            print(f"   RRF Score: {res['rrf_score']:.3f}")
            print(f"   BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}")
            print(f"   {res['description']}\n")
        else:
            print(f"  RRF Score: {res['rrf_score']:.3f}")
            print(f"  {res['description']}")

    if evaluate:
        import os
        from dotenv import load_dotenv
        from google import genai
        import json
        
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set")
            return
            
        client = genai.Client(api_key=api_key)
        
        formatted_results = []
        for res in results:
            formatted_results.append(f"{res.get('title', '')} - {res.get('description', '')}")
            
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )

        try:
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            scores = json.loads(clean_text.strip())
        except Exception:
            scores = []
            
        print()
        for i, res in enumerate(results, 1):
            score = scores[i - 1] if i - 1 < len(scores) else 0
            print(f"{i}. {res.get('title', '')}: {score}/3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search combining keyword and semantic scores")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic search vs BM25")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to use on results",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the results using an LLM",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            run_normalize(args.scores)
        case "weighted-search":
            run_weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            run_rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method, args.evaluate)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
