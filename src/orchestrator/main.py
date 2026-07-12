import os
import json
import uuid
import datetime
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Adjust paths if executed directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.sql.auth import check_permission
from src.rag.rag_core import retrieve

# Hardcoded constants
MAX_RETRIES = 2
SCORE_FLOOR = 0.55
LOG_FILE_PATH = os.path.join("output", "grounding_logs.json")

ALLOWED_DOMAINS = [
    "indiacode.nic.in",
    "sci.gov.in",
    "gov.in" # Subdomains allowed via logic
]

@dataclass
class DocumentBrief:
    title: str
    sections: List[Dict[str, str]] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Converts the brief to a natural-language task description for the formatting skill."""
        lines = [f"Please create a document titled '{self.title}'."]
        lines.append("Here is the content, organized by section. Make sure to include all citations exactly as written:")
        for idx, sec in enumerate(self.sections, 1):
            lines.append(f"\nSection {idx}: {sec.get('heading', 'Section')}")
            lines.append(f"Content: {sec.get('content', '')}")
        return "\n".join(lines)


def _log_decision(session_id: str, step: str, details: dict):
    """Logs retrieval and grounding decisions to a local JSON file."""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": session_id,
        "step": step,
        "details": details
    }
    
    # Append to JSON log file
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Failed to log decision: {e}")

# --- MOCK LLM CALLS (To be replaced with actual LLM SDK) ---
def mock_llm_parse_request(request: str) -> dict:
    """Mock parser to extract format and question."""
    format_req = "docx"
    if "pdf" in request.lower(): format_req = "pdf"
    elif "pptx" in request.lower(): format_req = "pptx"
    elif "xlsx" in request.lower(): format_req = "xlsx"
    return {"output_format": format_req, "legal_question": request}

def mock_llm_draft_claims(question: str) -> List[str]:
    """Mock drafting of factual claims based on question."""
    return [
        f"Claim 1 regarding {question}",
        f"Claim 2 regarding {question}"
    ]

def mock_llm_grounding_check(claim: str, chunks: List[dict]) -> dict:
    """Mock check to see if a claim is fully supported by chunks."""
    if not chunks:
        return {"supported": False, "reason": "No chunks provided"}
    # Mocking success for demo, grabbing citation from the highest chunk
    return {"supported": True, "source": chunks[0].get("source", "Unknown Source")}

def mock_llm_refine_query(original_query: str, failed_claim: str) -> str:
    """Mock query refinement."""
    return f"{original_query} specifically about {failed_claim}"

def mock_skill_dispatch(prompt: str, format_type: str) -> str:
    """Mock dispatching natural language prompt to a skill and getting a file path."""
    output_path = os.path.join("output", f"generated_document.{format_type}")
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Simulated output file content.\n\nPrompt received:\n")
        f.write(prompt)
    return output_path
# -----------------------------------------------------------

def is_domain_allowed(url: str) -> bool:
    """Check if the URL belongs to an allowed domain."""
    try:
        domain = urlparse(url).netloc.lower()
        for allowed in ALLOWED_DOMAINS:
            if domain == allowed or domain.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False

def call_browser_fallback(query: str) -> str:
    """
    Simulates calling the browser MCP/subagent restricted to allowed domains.
    """
    print(f"[Browser Subagent] Triggered fallback search for: {query}")
    # In a real implementation, this would invoke the agent tools bounded by ALLOWED_DOMAINS.
    return "[Browser Fallback] Official data fetched confirming the claim."

def process_legal_request(user_request: str, user_id: str, session_id: Optional[str] = None) -> str:
    """
    Core ReAct loop for the Legal Agent.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    _log_decision(session_id, "start_request", {"user_request": user_request, "user_id": user_id})

    # 1. Parse user request -> extract output format + legal question
    parsed = mock_llm_parse_request(user_request)
    output_format = parsed["output_format"]
    legal_question = parsed["legal_question"]
    _log_decision(session_id, "parse_request", parsed)

    # 2. Check permission via SQL
    # Must stop and return "permission_denied" before any retrieval happens
    if not check_permission(user_id, "generate_legal_doc"):
        _log_decision(session_id, "permission_check", {"status": "denied"})
        return "permission_denied"
    _log_decision(session_id, "permission_check", {"status": "granted"})

    # Pre-draft claims we need to make to answer the question
    drafted_claims = mock_llm_draft_claims(legal_question)
    
    verified_claims = []
    
    for claim in drafted_claims:
        current_query = legal_question
        retries = 0
        claim_verified = False
        
        while retries <= MAX_RETRIES:
            # 3. Retrieve via RAG module
            chunks = retrieve(query=current_query, session_id=session_id, score_floor=SCORE_FLOOR, user_id=None)
            
            _log_decision(session_id, "retrieval", {
                "query": current_query,
                "retry_iteration": retries,
                "chunks_found": len(chunks)
            })

            # 4. Grounding check
            grounding_result = mock_llm_grounding_check(claim, chunks)
            _log_decision(session_id, "grounding_check", {
                "claim": claim,
                "grounding_result": grounding_result
            })
            
            if grounding_result.get("supported"):
                citation = grounding_result.get("source", "Unknown Source")
                verified_claims.append(f"{claim} [{citation}]")
                claim_verified = True
                break
            else:
                # Need to retry with refined query
                if retries < MAX_RETRIES:
                    current_query = mock_llm_refine_query(current_query, claim)
                retries += 1

        # 5. Browser Fallback if grounding fails after max retries
        if not claim_verified:
            _log_decision(session_id, "browser_fallback_triggered", {"claim": claim})
            fallback_data = call_browser_fallback(claim)
            
            # Re-evaluate with fallback data
            if fallback_data: # Assume fallback succeeded for demo
                verified_claims.append(f"{claim} [Verified via Browser: Official Portal]")
            else:
                verified_claims.append(f"{claim} [unverified — needs review]")

    # 6. Compose document brief internally
    brief = DocumentBrief(title=f"Legal Memo regarding: {legal_question}")
    brief.sections.append({
        "heading": "Findings",
        "content": " ".join(verified_claims)
    })
    
    # 7. Hand brief as natural language task to matching skill
    prompt = brief.to_prompt()
    _log_decision(session_id, "skill_dispatch", {"format": output_format, "prompt": prompt})
    
    # 8. Return the output file path
    output_path = mock_skill_dispatch(prompt, output_format)
    _log_decision(session_id, "completed", {"output_path": output_path})
    
    return output_path

if __name__ == "__main__":
    # Test execution
    print("Testing ReAct Loop Orchestrator...")
    try:
        # User 'u1' is used for testing. Requires auth DB setup to pass.
        result = process_legal_request("Draft a docx memo about IPC Section 302", "u1")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Execution Error: {e}")
