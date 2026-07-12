import os
import json
import uuid
import datetime
import subprocess
import requests
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

class AmbiguousQueryError(Exception):
    pass

class NoMatchingLawError(Exception):
    pass

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

def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Make a real LLM call to Gemini API using the requests library."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Cannot make live LLM calls.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {}
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }
    payload["contents"] = [{
        "parts": [{"text": prompt}]
    }]

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        raise RuntimeError(f"LLM API Error: {response.status_code} - {response.text}")
        
    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected LLM response format: {data}") from e

def process_legal_request(user_request: str, user_id: str, session_id: Optional[str] = None) -> str:
    """
    Core ReAct loop for the Legal Agent.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        
    _log_decision(session_id, "start_request", {"user_request": user_request, "user_id": user_id})

    # 1. Parse user request -> extract output format + legal question
    parse_prompt = f"Extract output_format and legal_question from this request: '{user_request}'. Respond ONLY in JSON format: {{\"output_format\": \"...\", \"legal_question\": \"...\"}}. output_format should be one of docx, pdf, pptx, xlsx. If ambiguous, set legal_question to AMBIGUOUS."
    
    try:
        parsed_str = call_llm(parse_prompt)
        if "```json" in parsed_str:
            parsed_str = parsed_str.split("```json")[1].split("```")[0].strip()
        elif "```" in parsed_str:
            parsed_str = parsed_str.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(parsed_str)
        output_format = parsed.get("output_format", "docx").lower()
        legal_question = parsed.get("legal_question", user_request)
    except Exception:
        output_format = "docx"
        legal_question = user_request

    if legal_question == "AMBIGUOUS" or output_format not in ["docx", "pdf", "pptx", "xlsx"]:
        raise AmbiguousQueryError("Please clarify your query and the desired output format (docx, pdf, pptx, xlsx).")
        
    _log_decision(session_id, "parse_request", {"output_format": output_format, "legal_question": legal_question})

    # 2. Check permission via SQL
    if not check_permission(user_id, "generate_legal_doc"):
        _log_decision(session_id, "permission_check", {"status": "denied"})
        return "permission_denied"
    _log_decision(session_id, "permission_check", {"status": "granted"})

    verified_claims = []
    current_query = legal_question
    retries = 0
    claim_verified = False

    while retries <= MAX_RETRIES:
        # 3. Retrieve via RAG module FIRST
        chunks = retrieve(query=current_query, session_id=session_id, score_floor=SCORE_FLOOR, user_id=None)
        
        _log_decision(session_id, "retrieval", {
            "query": current_query,
            "retry_iteration": retries,
            "chunks_found": len(chunks)
        })
        
        if not chunks:
            if retries < MAX_RETRIES:
                refine_prompt = f"The query '{current_query}' returned no relevant results. Refine the query to be more generic or use synonyms to improve search results."
                current_query = call_llm(refine_prompt)
                retries += 1
                continue
            else:
                raise NoMatchingLawError("No matching law found in the corpus.")

        # 4. Draft claims grounded in the retrieved chunks
        context = "\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in chunks])
        draft_prompt = f"Based ONLY on the following context, answer the legal question: '{legal_question}'. List the factual claims as bullet points. If the context does not contain the answer, say 'UNSUPPORTED'.\n\nContext:\n{context}"
        
        drafted_str = call_llm(draft_prompt)
        
        if "UNSUPPORTED" in drafted_str.upper():
            if retries < MAX_RETRIES:
                refine_prompt = f"The query '{current_query}' did not yield the exact answer. Refine it based on what might be missing."
                current_query = call_llm(refine_prompt)
                retries += 1
                continue
            else:
                raise NoMatchingLawError("No matching law found in the corpus to support the question.")
                
        # 5. Grounding check
        claims = [c.strip("-* ") for c in drafted_str.split("\n") if c.strip().startswith("-") or c.strip().startswith("*")]
        if not claims:
            claims = [drafted_str.strip()]
            
        all_grounded = True
        
        for claim in claims:
            ground_prompt = f"Verify if the following claim is fully supported by the provided context. If it is, return JSON: {{\"supported\": true, \"source\": \"<name of source>\"}}. If not, return JSON: {{\"supported\": false, \"reason\": \"<why it failed>\"}}.\n\nClaim: {claim}\n\nContext:\n{context}"
            
            try:
                grounding_res_str = call_llm(ground_prompt)
                
                if "```json" in grounding_res_str:
                    grounding_res_str = grounding_res_str.split("```json")[1].split("```")[0].strip()
                elif "```" in grounding_res_str:
                    grounding_res_str = grounding_res_str.split("```")[1].split("```")[0].strip()
                
                grounding_result = json.loads(grounding_res_str)
            except Exception:
                if "true" in grounding_res_str.lower():
                    grounding_result = {"supported": True, "source": chunks[0]["source"] if chunks else "Unknown Source"}
                else:
                    grounding_result = {"supported": False, "reason": "Failed to parse json"}

            _log_decision(session_id, "grounding_check", {
                "claim": claim,
                "grounding_result": grounding_result
            })
            
            if grounding_result.get("supported"):
                citation = grounding_result.get("source", "Unknown Source")
                verified_claims.append(f"{claim} [{citation}]")
            else:
                all_grounded = False
                verified_claims.append(f"{claim} [unverified — needs review]")
                break

        if all_grounded and verified_claims:
            claim_verified = True
            break
        else:
            if retries < MAX_RETRIES:
                failed_claim = claims[0] if claims else ""
                refine_prompt = f"The claim '{failed_claim}' was not fully supported by the context. Generate a new search query to find the missing information."
                current_query = call_llm(refine_prompt)
                verified_claims = []
            retries += 1
            
    if not claim_verified and not verified_claims:
        raise NoMatchingLawError("No matching law found in the corpus.")

    # 6. Compose document brief
    brief = DocumentBrief(title=f"Legal Memo regarding: {legal_question}")
    brief.sections.append({
        "heading": "Findings",
        "content": " ".join(verified_claims)
    })
    
    prompt = brief.to_prompt()
    _log_decision(session_id, "skill_dispatch", {"format": output_format, "prompt": prompt})

    # 7. Real Skill Dispatch
    skill_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".gemini", "skills", output_format, "SKILL.md"))
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            skill_content = f.read()
    except Exception:
        skill_content = f"Generate a script to create a {output_format} file."
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"generated_document.{output_format}")
    
    script_type = "javascript" if output_format in ["docx", "pptx"] else "python"
    script_ext = "js" if script_type == "javascript" else "py"
    
    # Path with double slashes for windows compatibility in generated script
    out_file_escaped = out_file.replace('\\', '\\\\')
    
    dispatch_prompt = f"Write a {script_type} script that creates a document matching this prompt:\n\n{prompt}\n\nThe script MUST save the file exactly to: '{out_file_escaped}'.\n\nReturn ONLY the raw code inside a markdown code block (e.g. ```{script_type} ... ```). Do not include any other text."
    
    script_content = call_llm(dispatch_prompt, system_prompt=skill_content)
    
    if f"```{script_type}" in script_content:
        script_code = script_content.split(f"```{script_type}")[1].split("```")[0].strip()
    elif "```" in script_content:
        script_code = script_content.split("```")[1].split("```")[0].strip()
    else:
        script_code = script_content.strip()
        
    script_path = os.path.join(out_dir, f"run_skill_{session_id}.{script_ext}")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_code)
        
    if script_type == "javascript":
        subprocess.run(["node", script_path], check=True)
    else:
        subprocess.run(["python", script_path], check=True)
        
    _log_decision(session_id, "completed", {"output_path": out_file})
    
    return out_file

if __name__ == "__main__":
    print("Testing ReAct Loop Orchestrator...")
    try:
        result = process_legal_request("Draft a docx memo about IPC Section 302", "u1")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Execution Error: {e}")
