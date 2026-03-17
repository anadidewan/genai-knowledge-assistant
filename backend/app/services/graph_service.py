import json
from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def extract_graph_from_chunk(chunk: dict) -> dict:
    prompt = f"""
You are an information extraction system.

From the following text, extract:
1. Important entities
2. Relationships between entities

Return ONLY valid JSON in this format:

{{
  "entities": ["entity1", "entity2"],
  "relationships": [
    {{
      "source": "entity1",
      "relation": "related_to",
      "target": "entity2"
    }}
  ]
}}

Rules:
- Keep entities short and meaningful
- Only use information present in the text
- If no clear relationships exist, return empty lists
- Do not include any explanation outside the JSON

Text:
{chunk['text']}
"""

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt
    )

    raw_text = response.text.strip()
    raw_text = response.text.strip()
    print(f"\n--- GRAPH RAW OUTPUT chunk {chunk['chunk_id']} ---")
    print(raw_text)

    cleaned = raw_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        print(f"JSON parse failed for chunk {chunk['chunk_id']}: {e}")
        parsed = {
            "entities": [],
            "relationships": []
        }
    

    return {
        "document_name": chunk["document_name"],
        "chunk_id": chunk["chunk_id"],
        "entities": parsed.get("entities", []),
        "relationships": parsed.get("relationships", [])
    }


def build_graph_data(chunks: list[dict]) -> list[dict]:
    graph_records = []

    for chunk in chunks:
        graph_record = extract_graph_from_chunk(chunk)
        graph_records.append(graph_record)

    return graph_records