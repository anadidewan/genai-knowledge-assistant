from prometheus_client import Counter, Histogram, Gauge

# Total chat requests
CHAT_REQUESTS_TOTAL = Counter(
    "rag_chat_requests_total",
    "Total number of chat requests received"
)

# Total chat request failures
CHAT_REQUEST_FAILURES_TOTAL = Counter(
    "rag_chat_request_failures_total",
    "Total number of failed chat requests"
)

# Request latency
CHAT_REQUEST_LATENCY_MS = Histogram(
    "rag_chat_request_latency_ms",
    "End-to-end chat request latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000)
)

# Routing decisions
ROUTING_DECISIONS_TOTAL = Counter(
    "rag_routing_decisions_total",
    "Count of routing decisions",
    ["mode"]  # direct, retrieve, critique
)

# Retrieval latency
RETRIEVAL_LATENCY_MS = Histogram(
    "rag_retrieval_latency_ms",
    "Retrieval latency in milliseconds",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000)
)

# Retrieval confidence
RETRIEVAL_CONFIDENCE = Histogram(
    "rag_retrieval_confidence",
    "Distribution of retrieval confidence",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# Retrieved chunks count
RETRIEVED_CHUNKS_COUNT = Histogram(
    "rag_retrieved_chunks_count",
    "Number of retrieved chunks returned",
    buckets=(0, 1, 2, 3, 5, 8, 10)
)

# Low confidence fallbacks
LOW_CONFIDENCE_FALLBACKS_TOTAL = Counter(
    "rag_low_confidence_fallbacks_total",
    "Number of times the system fell back to direct answer because retrieval confidence was low"
)

# LLM calls
LLM_CALLS_TOTAL = Counter(
    "rag_llm_calls_total",
    "Total LLM calls made",
    ["caller"]  # generate_answer, generate_direct_answer, generate_critique_answer, rewrite_and_route
)

# LLM latency
LLM_LATENCY_MS = Histogram(
    "rag_llm_latency_ms",
    "LLM call latency in milliseconds",
    ["caller"],
    buckets=(50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000)
)

# Empty LLM responses
LLM_EMPTY_RESPONSES_TOTAL = Counter(
    "rag_llm_empty_responses_total",
    "Number of empty responses returned by the LLM",
    ["caller"]
)

# Retry count
LLM_RETRIES_TOTAL = Counter(
    "rag_llm_retries_total",
    "Total retry attempts for LLM calls"
)