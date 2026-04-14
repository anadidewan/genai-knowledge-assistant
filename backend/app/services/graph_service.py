import spacy
from itertools import combinations
from app.utils.custom_logger import get_logger

logger = get_logger(__name__)

# ── Load spaCy model once at import ────────────────────────────────
# Install the model with:  python -m spacy download en_core_web_sm
# For better accuracy use:  python -m spacy download en_core_web_trf
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded: en_core_web_sm")
except OSError:
    logger.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Install it with: python -m spacy download en_core_web_sm"
    )
    raise
# ────────────────────────────────────────────────────────────────────


def _find_root_verb(token) -> str | None:
    """Walk up the dependency tree from a token to find the nearest verb ancestor."""
    current = token
    seen = set()
    while current.head != current and current.dep_ != "ROOT":
        if current.i in seen:
            break
        seen.add(current.i)
        current = current.head
        if current.pos_ == "VERB":
            return current.lemma_
    return None


def _extract_dep_relation(ent_a, ent_b, sent) -> str | None:
    """
    Try to find a verb connecting two entities in the same sentence
    by walking the dependency tree from each entity's root token.
    """
    a_root = ent_a.root
    b_root = ent_b.root

    # Case 1: both entities share the same head verb
    a_verb = _find_root_verb(a_root)
    b_verb = _find_root_verb(b_root)

    if a_verb and a_verb == b_verb:
        return a_verb

    # Case 2: one entity is the subject and the other is the object of the same verb
    if a_root.head == b_root.head and a_root.head.pos_ == "VERB":
        return a_root.head.lemma_

    # Case 3: direct prep relationship  (e.g. "CEO of Google")
    if a_root.head == b_root or b_root.head == a_root:
        bridge = a_root.head if a_root.head == b_root else b_root.head
        if bridge.pos_ in ("ADP", "VERB"):
            return bridge.lemma_

    return None


def extract_graph_from_chunk(chunk: dict) -> dict:
    """
    Extract entities and relationships from a single chunk using
    spaCy NER + dependency parsing, with co-occurrence fallback.
    """
    text = chunk["text"]
    doc = nlp(text)

    # ── Step 1: Collect unique named entities ──────────────────────
    raw_entities = {}
    for ent in doc.ents:
        # Normalize whitespace and skip very short / noisy entities
        clean = " ".join(ent.text.split())
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key not in raw_entities:
            raw_entities[key] = {
                "text": clean,
                "label": ent.label_,
            }

    entity_names = list(raw_entities.keys())
    entities = [raw_entities[k]["text"] for k in entity_names]

    # ── Step 2: Extract relationships ──────────────────────────────
    relationships = []
    seen_pairs = set()

    for sent in doc.sents:
        # Get entities that appear in this sentence
        sent_ents = [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]

        for ent_a, ent_b in combinations(sent_ents, 2):
            a_key = " ".join(ent_a.text.split()).lower()
            b_key = " ".join(ent_b.text.split()).lower()

            if a_key == b_key:
                continue

            pair = tuple(sorted([a_key, b_key]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Try dependency-based relation first
            relation = _extract_dep_relation(ent_a, ent_b, sent)

            # Fallback: co-occurrence in the same sentence
            if not relation:
                relation = "related_to"

            relationships.append({
                "source": raw_entities[a_key]["text"],
                "relation": relation,
                "target": raw_entities[b_key]["text"],
            })

    logger.debug(
        "Graph extracted | chunk=%s | entities=%d | relationships=%d",
        chunk["chunk_id"],
        len(entities),
        len(relationships),
    )

    return {
        "document_name": chunk["document_name"],
        "chunk_id": chunk["chunk_id"],
        "entities": entities,
        "relationships": relationships,
    }


def build_graph_data(chunks: list[dict]) -> list[dict]:
    logger.info("Graph extraction starting | chunks=%d", len(chunks))

    graph_records = []
    for chunk in chunks:
        graph_record = extract_graph_from_chunk(chunk)
        graph_records.append(graph_record)

    total_entities = sum(len(r["entities"]) for r in graph_records)
    total_rels = sum(len(r["relationships"]) for r in graph_records)
    logger.info(
        "Graph extraction complete | chunks=%d | total_entities=%d | total_relationships=%d",
        len(chunks),
        total_entities,
        total_rels,
    )

    return graph_records