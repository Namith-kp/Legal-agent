import os
import sys
import json
import uuid
import shutil
import tempfile
import sqlite3
import pytest
from unittest.mock import patch, MagicMock

# Add workspace root to sys.path so we can import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.orchestrator.main import process_legal_request, AmbiguousQueryError, NoMatchingLawError
import src.orchestrator.main as orchestrator_module
from src.rag import ingest_law_corpus, retrieve
import src.rag.rag_core as rag_core

def generate_simple_pdf(text: str) -> bytes:
    """Generates a minimal PDF stream that pypdf can extract text from."""
    escaped_text = text.replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 100 700 Td ({escaped_text}) Tj ET"
    length = len(stream)
    pdf_content = (
        f"%PDF-1.4\n"
        f"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        f"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        f"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        f"5 0 obj\n<< /Length {length} >>\nstream\n{stream}\nendstream\nendobj\n"
        f"xref\n0 6\n0000000000 65535 f\n"
        f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n300\n%%EOF\n"
    )
    return pdf_content.encode('utf-8')

@pytest.fixture(scope="function")
def setup_e2e():
    """Sets up temporary database, Chroma DB, and mocks for E2E tests."""
    # 1. Setup temporary SQLite database
    db_fd, db_path = tempfile.mkstemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    
    # Run migrations using the schema.sql file
    schema_path = os.path.join(parent_dir, "src", "sql", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        cursor = conn.cursor()
        
        # Seed roles
        cursor.execute("INSERT INTO roles (id, name) VALUES (?, ?)", ("role-admin", "admin"))
        cursor.execute("INSERT INTO roles (id, name) VALUES (?, ?)", ("role-viewer", "viewer"))
        
        # Seed permissions
        cursor.execute("INSERT INTO permissions (id, name) VALUES (?, ?)", ("perm-generate", "generate_legal_doc"))
        
        # Seed role_permissions
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-admin", "perm-generate"))
        
        # Seed users
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-admin", "Admin User", "role-admin"))
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-denied", "Denied User", "role-viewer"))
        conn.commit()
    finally:
        conn.close()

    # 2. Setup temporary Chroma DB
    chroma_temp_dir = tempfile.mkdtemp()
    os.environ["CHROMA_DB_DIR"] = chroma_temp_dir
    old_chroma_dir = rag_core.CHROMA_DB_DIR
    rag_core.CHROMA_DB_DIR = chroma_temp_dir
    
    # 3. Create mock law corpus documents
    corpus_dir = tempfile.mkdtemp()
    
    speeding_pdf = os.path.join(corpus_dir, "speeding_law.pdf")
    with open(speeding_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 183 details penalties for speeding. "
            "Speeding results in a fine of Rs 1000 for light motor vehicles."
        ))
        
    drunk_driving_pdf = os.path.join(corpus_dir, "drunk_driving_law.pdf")
    with open(drunk_driving_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 185 details penalties for drunk driving. "
            "Drunk driving results in a fine of Rs 10000 or imprisonment up to 6 months."
        ))
        
    no_license_pdf = os.path.join(corpus_dir, "no_license_law.pdf")
    with open(no_license_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 181 details penalties for driving without a license. "
            "Driving without a license results in a fine of Rs 5000 or imprisonment up to 3 months."
        ))
        
    ingest_law_corpus(corpus_dir)
    
    # 4. Results folder setup
    results_dir = os.path.join(parent_dir, "tests", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Patch LOG_FILE_PATH
    old_log_path = orchestrator_module.LOG_FILE_PATH
    orchestrator_module.LOG_FILE_PATH = os.path.join(results_dir, "grounding_logs.json")
    
    # Create output dir for scripts
    out_dir = os.path.join(parent_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    yield results_dir
    
    # Clean up DB
    os.close(db_fd)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
            
    # Clean up Chroma & Corpus
    rag_core.CHROMA_DB_DIR = old_chroma_dir
    shutil.rmtree(chroma_temp_dir, ignore_errors=True)
    shutil.rmtree(corpus_dir, ignore_errors=True)
    
    # Restore log path
    orchestrator_module.LOG_FILE_PATH = old_log_path
    
    # Clean up env
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]
    if "CHROMA_DB_DIR" in os.environ:
        del os.environ["CHROMA_DB_DIR"]


# --- MOCK HELPERS ---
def custom_call_llm(prompt: str, system_prompt: str = None) -> str:
    prompt_lower = prompt.lower()
    
    # 1. Parse Request
    if "extract output_format" in prompt_lower:
        if "driving." in prompt_lower:
            return '```json\n{"output_format": "docx", "legal_question": "AMBIGUOUS"}\n```'
        if "martian" in prompt_lower:
            return '```json\n{"output_format": "docx", "legal_question": "Martian space traffic penalties"}\n```'
        if "compare penalties for drunk driving" in prompt_lower:
            return '```json\n{"output_format": "xlsx", "legal_question": "drunk driving and driving without a license penalties"}\n```'
        if "section 183 and section 185" in prompt_lower:
            return '```json\n{"output_format": "docx", "legal_question": "Section 183 and 185"}\n```'
        return '```json\n{"output_format": "docx", "legal_question": "Section 183 speeding penalties"}\n```'
        
    # 2. Refine Query
    if "refine" in prompt_lower or "new search query" in prompt_lower:
        if "martian" in prompt_lower:
            return "alien laws"
        return "refined generic query"
        
    # 3. Draft claims
    if "list the factual claims as bullet points" in prompt_lower:
        if "martian" in prompt_lower:
            return "UNSUPPORTED"
        if "183 and 185" in prompt_lower:
            return "- Section 183 details speeding penalties\n- Section 185 details drunk driving penalties"
        if "drunk driving and driving without a license" in prompt_lower:
            return "- Section 185 details penalties for drunk driving\n- Section 181 details penalties for driving without a license"
        return "- Section 183 details penalties for speeding"
        
    # 4. Grounding verification
    if "verify if the following claim is fully supported" in prompt_lower:
        if "section 999" in prompt_lower or "martian" in prompt_lower:
            return '```json\n{"supported": false, "reason": "Not in context"}\n```'
        return '```json\n{"supported": true, "source": "test_source.pdf"}\n```'
        
    # 5. Skill dispatch script generation
    if "write a javascript script" in prompt_lower:
        return "```javascript\nconsole.log('mock');\n```"
    if "write a python script" in prompt_lower:
        return "```python\nprint('mock')\n```"
        
    return ""

def custom_subprocess_run(args, **kwargs):
    # Find the output path expected by reading the prompt or just hardcode based on test.
    # main.py expects generated_document.<format> in output/
    # We can create a dummy file for the tests to succeed.
    script_path = args[1]
    out_dir = os.path.dirname(script_path)
    
    for fmt in ["docx", "pdf", "pptx", "xlsx"]:
        path = os.path.join(out_dir, f"generated_document.{fmt}")
        with open(path, "w") as f:
            f.write(f"Mock {fmt} output")
            
    return MagicMock()
# --------------------


def test_permission_denied(setup_e2e):
    """Test 1: Confirms that a user without generate_legal_doc permission is denied immediately."""
    user_id = "user-denied"
    query = "Draft a docx memo about Section 183 speeding penalties."
    session_id = str(uuid.uuid4())
    
    result = process_legal_request(query, user_id, session_id=session_id)
    assert result == "permission_denied"
    
    log_file = os.path.join(setup_e2e, "grounding_logs.json")
    with open(log_file, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f if line.strip()]
        
    logs = [log for log in logs if log["session_id"] == session_id]
    steps = [log["step"] for log in logs]
    
    assert "start_request" in steps
    assert "parse_request" in steps
    assert "permission_check" in steps
    assert "retrieval" not in steps


def test_single_clean_offense_docx(setup_e2e):
    """Test 2: A single clean offense query (traffic offense penalties, docx output) should succeed."""
    user_id = "user-admin"
    query = "Draft a docx memo about Motor Vehicles Act Section 183 speeding penalties."
    session_id = str(uuid.uuid4())
    
    with patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm), \
         patch("subprocess.run", side_effect=custom_subprocess_run):
         
        output_path = process_legal_request(query, user_id, session_id=session_id)
        
        target_path = os.path.join(setup_e2e, "traffic_speeding_penalties.docx")
        shutil.copy(output_path, target_path)
        assert os.path.exists(target_path)


def test_multi_offense_comparison_xlsx(setup_e2e):
    """Test 3: A multi-offense comparison (xlsx output) should succeed."""
    user_id = "user-admin"
    query = "Compare penalties for drunk driving and driving without a license in xlsx format."
    session_id = str(uuid.uuid4())
    
    with patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm), \
         patch("subprocess.run", side_effect=custom_subprocess_run):
         
        output_path = process_legal_request(query, user_id, session_id=session_id)
        
        target_path = os.path.join(setup_e2e, "multi_offense_comparison.xlsx")
        shutil.copy(output_path, target_path)
        assert os.path.exists(target_path)


def test_ambiguous_query_clarification(setup_e2e):
    """Test 4: An ambiguous query should trigger a clarifying question instead of a guess."""
    user_id = "user-admin"
    query = "Draft a docx memo about penalties for driving."
    session_id = str(uuid.uuid4())
    
    with patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm):
        with pytest.raises(AmbiguousQueryError) as exc_info:
            process_legal_request(query, user_id, session_id=session_id)
            
        assert "clarify" in str(exc_info.value).lower()


def test_no_matching_law_refusal(setup_e2e):
    """Test 5: A query with no matching law in the corpus should refuse to answer instead of hallucinating."""
    user_id = "user-admin"
    query = "Draft a docx memo about Martian space traffic penalties."
    session_id = str(uuid.uuid4())
    
    with patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm):
        with pytest.raises(NoMatchingLawError) as exc_info:
            process_legal_request(query, user_id, session_id=session_id)


def test_rag_score_filtering(setup_e2e):
    """Test 6: Verify that chunks below the score floor of 0.55 are filtered out."""
    results = retrieve("unrelated random text about bananas and monkeys", score_floor=0.55)
    assert len(results) == 0


def test_max_retries_and_query_refinement(setup_e2e):
    """Test 7: Verify that grounding failures trigger query refinement up to MAX_RETRIES (2)."""
    user_id = "user-admin"
    query = "Draft a docx memo about Section 183."
    session_id = str(uuid.uuid4())
    
    retrieved_queries = []
    
    def custom_retrieve(query, session_id=None, score_floor=0.55, user_id=None):
        retrieved_queries.append(query)
        return []

    with patch("src.orchestrator.main.retrieve", side_effect=custom_retrieve), \
         patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm):
         
        with pytest.raises(NoMatchingLawError):
            process_legal_request(query, user_id, session_id=session_id)
            
        assert len(retrieved_queries) == 3


def test_multiple_claims_grounded(setup_e2e):
    """Test 8: Verify that multiple drafted claims are grounded individually."""
    user_id = "user-admin"
    query = "Draft a docx memo about Section 183 and Section 185"
    session_id = str(uuid.uuid4())
    
    def custom_retrieve(query, session_id=None, score_floor=0.55, user_id=None):
        return [{"source": "test_source.pdf", "text": "Mock content for 183 and 185", "score": 0.99}]
    
    with patch("src.orchestrator.main.call_llm", side_effect=custom_call_llm), \
         patch("src.orchestrator.main.retrieve", side_effect=custom_retrieve), \
         patch("subprocess.run", side_effect=custom_subprocess_run):
         
        process_legal_request(query, user_id, session_id=session_id)
        
        log_file = os.path.join(setup_e2e, "grounding_logs.json")
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            
        session_logs = [log for log in logs if log["session_id"] == session_id]
        grounding_logs = [log for log in session_logs if log["step"] == "grounding_check"]
        
        assert len(grounding_logs) >= 2
        claims_checked = [g["details"]["claim"] for g in grounding_logs]
        
        assert any("183" in c for c in claims_checked)
        assert any("185" in c for c in claims_checked)
