#!/usr/bin/env python3
import argparse
import mimetypes
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():
    parser = argparse.ArgumentParser(description="Multimodal query rewriting using Gemini.")
    parser.add_argument("--image", required=True, help="path to an image file")
    parser.add_argument("--query", required=True, help="a text query to rewrite based on the image")
    
    args = parser.parse_args()
    
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    
    try:
        with open(args.image, "rb") as f:
            img = f.read()
    except FileNotFoundError:
        print(f"Error: Profile image '{args.image}' not found.")
        return

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return

    client = genai.Client(api_key=api_key)
    
    system_prompt = (
        "Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:\n"
        "- Synthesize visual and textual information\n"
        "- Focus on movie-specific details (actors, scenes, style, etc.)\n"
        "- Return only the rewritten query, without any additional commentary"
    )
    
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parts,
    )
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()
