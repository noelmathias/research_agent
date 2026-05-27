from pydantic import BaseModel
from typing import Optional, List, Dict, Any


# ── Health ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_reachable: bool
    model: str


class OllamaTestRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


class OllamaTestResponse(BaseModel):
    response: str
    model: str


# ── Research / Planner ────────────────────────────────────────────────────────
class ResearchPlanRequest(BaseModel):
    query: str
    model: Optional[str] = None


class ResearchSubtask(BaseModel):
    step: int
    task: str
    reason: str


class ResearchPlanResponse(BaseModel):
    query: str
    plan: List[ResearchSubtask]
    raw_plan: str
    model: str


# ── Research / Full Run ───────────────────────────────────────────────────────
class ResearchRunRequest(BaseModel):
    query: str
    model: Optional[str] = None
    collection_name: Optional[str] = None


class RetrievalInfo(BaseModel):
    chunks_retrieved: int
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]


class EvaluationInfo(BaseModel):
    confidence_score: float
    passed: bool
    hallucination_flags: List[str]
    reasoning: str
    retry_count: int
    reflection_notes: str


class ReportInfo(BaseModel):
    report_id: str
    citations: List[Dict[str, Any]]
    final_report: str


class ResearchRunResponse(BaseModel):
    query: str
    model: str
    plan: List[ResearchSubtask]
    retrieval: RetrievalInfo
    summary: str
    evaluation: EvaluationInfo
    report: ReportInfo
    current_step: str
    error: Optional[str] = None


class ResearchJobAccepted(BaseModel):
    job_id: str
    status: str


class ResearchJobEvent(BaseModel):
    job_id: str
    event_id: str
    event_type: str
    node: str
    status: str
    message: str
    timestamp: str
    payload: Dict[str, Any] = {}


class ResearchJobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    model: str
    collection_name: Optional[str] = None
    created_at: str
    updated_at: str
    current_step: str
    error: Optional[str] = None
    events_emitted: int
    result: Optional[ResearchRunResponse] = None


# ── Vector / Ingestion ────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    collection_name: Optional[str] = None


class IngestResponse(BaseModel):
    collection: str
    chunks_stored: int
    ids: List[str]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    collection: str
    results: List[SearchResult]
    total_results: int

class VectorStoreStatusResponse(BaseModel):
    collection_name: str
    document_count: int
    embedding_model: str

# ── PDF ───────────────────────────────────────────────────────────────────────
class PDFPageInfo(BaseModel):
    page_number: int
    char_count: int
    word_count: int


class PDFUploadResponse(BaseModel):
    filename: str
    file_path: str
    total_pages: int
    total_chars: int
    chunks_stored: int
    collection: str
    page_info: List[PDFPageInfo]
    ids: List[str]


class PDFParseResult(BaseModel):
    filename: str
    total_pages: int
    total_chars: int
    pages: List[Dict[str, Any]]


# ── Async Jobs ────────────────────────────────────────────────────────────────

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    stream_url: str
    status_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    query: str
    model: str
    created_at: str
    updated_at: str
    event_count: int
    has_result: bool
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
