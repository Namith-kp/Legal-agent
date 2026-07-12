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

from src.orchestrator.main import process_legal_request, DocumentBrief
import src.orchestrator.main as orchestrator_module
from src.rag import ingest_law_corpus, retrieve
import src.rag.rag_core as rag_core
from src.sql.auth import check_permission

# Custom exceptions for E2E tests to simulate correct LLM error handling
class AmbiguousQueryError(Exception):
    """Raised when the LLM/Orchestrator detects that a query is ambiguous and needs clarification."""
    pass

class NoMatchingLawError(Exception):
    """Raised when the LLM/Orchestrator refuses to generate a document because no matching law is found."""
    pass

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
    
    # Speeding law document
    speeding_pdf = os.path.join(corpus_dir, "speeding_law.pdf")
    with open(speeding_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 183 details penalties for speeding. "
            "Speeding results in a fine of Rs 1000 for light motor vehicles."
        ))
        
    # Drunk driving law document
    drunk_driving_pdf = os.path.join(corpus_dir, "drunk_driving_law.pdf")
    with open(drunk_driving_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 185 details penalties for drunk driving. "
            "Drunk driving results in a fine of Rs 10000 or imprisonment up to 6 months."
        ))
        
    # Driving without license law document
    no_license_pdf = os.path.join(corpus_dir, "no_license_law.pdf")
    with open(no_license_pdf, "wb") as f:
        f.write(generate_simple_pdf(
            "Motor Vehicles Act, 1988. Section 181 details penalties for driving without a license. "
            "Driving without a license results in a fine of Rs 5000 or imprisonment up to 3 months."
        ))
        
    # Ingest the mock law corpus
    ingest_law_corpus(corpus_dir)
    
    # 4. Results folder setup
    results_dir = os.path.join(parent_dir, "tests", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Patch LOG_FILE_PATH
    old_log_path = orchestrator_module.LOG_FILE_PATH
    orchestrator_module.LOG_FILE_PATH = os.path.join(results_dir, "grounding_logs.json")
    
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
    
    # Clean up environment variables
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]
    if "CHROMA_DB_DIR" in os.environ:
        del os.environ["CHROMA_DB_DIR"]


def test_permission_denied(setup_e2e):
    """Test 1: Confirms that a user without generate_legal_doc permission is denied immediately."""
    user_id = "user-denied"
    query = "Draft a docx memo about Section 183 speeding penalties."
    session_id = str(uuid.uuid4())
    
    result = process_legal_request(query, user_id, session_id=session_id)
    assert result == "permission_denied"
    
    # Check that grounding_logs has permission_check: denied and no retrieval
    log_file = os.path.join(setup_e2e, "grounding_logs.json")
    assert os.path.exists(log_file)
    
    with open(log_file, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f if line.strip()]
        
    logs = [log for log in logs if log["session_id"] == session_id]
        
    # Verify events
    steps = [log["step"] for log in logs]
    assert "start_request" in steps
    assert "parse_request" in steps
    assert "permission_check" in steps
    assert "retrieval" not in steps
    
    permission_log = next(log for log in logs if log["step"] == "permission_check")
    assert permission_log["details"]["status"] == "denied"


def test_single_clean_offense_docx(setup_e2e):
    """Test 2: A single clean offense query (traffic offense penalties, docx output) should succeed."""
    user_id = "user-admin"
    query = "Draft a docx memo about Motor Vehicles Act Section 183 speeding penalties."
    session_id = str(uuid.uuid4())
    
    # Custom parser to extract format and legal question
    def custom_parse(req):
        return {"output_format": "docx", "legal_question": "Section 183 speeding penalties"}
        
    # Custom claims drafting
    def custom_draft(q):
        return ["Section 183 details penalties for speeding"]
        
    # Custom grounding check
    def custom_grounding(claim, chunks):
        # Ensure RAG actually found matching chunks
        assert len(chunks) > 0
        assert "Section 183" in chunks[0]["text"]
        return {"supported": True, "source": chunks[0]["source"]}

    # Custom skill dispatch writing output to tests/results/
    def custom_skill_dispatch(prompt, format_type):
        out_path = os.path.join(setup_e2e, f"generated_document.{format_type}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Draft Document\n\nPrompt: {prompt}")
        return out_path

    with patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft), \
         patch("src.orchestrator.main.mock_llm_grounding_check", side_effect=custom_grounding), \
         patch("src.orchestrator.main.mock_skill_dispatch", side_effect=custom_skill_dispatch):
         
        output_path = process_legal_request(query, user_id, session_id=session_id)
        
        # Ensure correct output path returned
        target_path = os.path.join(setup_e2e, "traffic_speeding_penalties.docx")
        shutil.copy(output_path, target_path)
        
        assert os.path.exists(target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Section 183 details penalties for speeding" in content
            assert "Motor Vehicles Act" in content


def test_multi_offense_comparison_xlsx(setup_e2e):
    """Test 3: A multi-offense comparison (xlsx output) should succeed."""
    user_id = "user-admin"
    query = "Compare penalties for drunk driving and driving without a license in xlsx format."
    session_id = str(uuid.uuid4())
    
    def custom_parse(req):
        return {"output_format": "xlsx", "legal_question": "drunk driving and driving without a license penalties"}
        
    def custom_draft(q):
        # We draft two separate claims to verify
        return [
            "Section 185 details penalties for drunk driving",
            "Section 181 details penalties for driving without a license"
        ]
        
    def custom_grounding(claim, chunks):
        assert len(chunks) > 0
        # Verify the chunks correspond to the claim being validated
        if "drunk driving" in claim.lower() or "185" in claim:
            assert any("185" in c["text"] for c in chunks)
        elif "license" in claim.lower() or "181" in claim:
            assert any("181" in c["text"] for c in chunks)
        return {"supported": True, "source": chunks[0]["source"]}

    def custom_skill_dispatch(prompt, format_type):
        out_path = os.path.join(setup_e2e, f"generated_document.{format_type}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Draft Spreadsheet\n\nPrompt: {prompt}")
        return out_path

    with patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft), \
         patch("src.orchestrator.main.mock_llm_grounding_check", side_effect=custom_grounding), \
         patch("src.orchestrator.main.mock_skill_dispatch", side_effect=custom_skill_dispatch):
         
        output_path = process_legal_request(query, user_id, session_id=session_id)
        
        target_path = os.path.join(setup_e2e, "multi_offense_comparison.xlsx")
        shutil.copy(output_path, target_path)
        
        assert os.path.exists(target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "drunk driving" in content.lower()
            assert "license" in content.lower()


def test_ambiguous_query_clarification(setup_e2e):
    """Test 4: An ambiguous query should trigger a clarifying question instead of a guess."""
    user_id = "user-admin"
    query = "Draft a docx memo about penalties for driving."
    session_id = str(uuid.uuid4())
    
    # Custom parser that identifies ambiguity and raises a clarification exception
    def custom_parse(req):
        if "driving" in req.lower() and not ("speeding" in req.lower() or "drunk" in req.lower() or "license" in req.lower()):
            raise AmbiguousQueryError(
                "Please clarify which traffic offense you are referring to: speeding, drunk driving, or driving without a license?"
            )
        return {"output_format": "docx", "legal_question": req}

    with patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse):
        with pytest.raises(AmbiguousQueryError) as exc_info:
            process_legal_request(query, user_id, session_id=session_id)
            
        assert "Please clarify which traffic offense you are referring to" in str(exc_info.value)


def test_no_matching_law_refusal(setup_e2e):
    """Test 5: A query with no matching law in the corpus should refuse to answer instead of hallucinating."""
    user_id = "user-admin"
    query = "Draft a docx memo about Martian space traffic penalties."
    session_id = str(uuid.uuid4())
    
    def custom_parse(req):
        return {"output_format": "docx", "legal_question": "Martian space traffic penalties"}
        
    def custom_draft(q):
        return ["Martian space traffic penalties are punishable by 100 space credits"]
        
    # Browser fallback raises refusal exception because there's no matching law anywhere
    def custom_browser_fallback(claim):
        raise NoMatchingLawError(
            "No matching law found in the corpus or online regarding Martian space traffic penalties."
        )

    with patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft), \
         patch("src.orchestrator.main.call_browser_fallback", side_effect=custom_browser_fallback):
         
        with pytest.raises(NoMatchingLawError) as exc_info:
            process_legal_request(query, user_id, session_id=session_id)
            
        assert "No matching law found" in str(exc_info.value)


def test_rag_score_filtering(setup_e2e):
    """Test 6: Verify that chunks below the score floor of 0.55 are filtered out."""
    user_id = "user-admin"
    query = "Draft a docx memo about something unrelated to traffic laws."
    
    # Call retrieve directly with a totally unrelated query
    # Since only traffic laws are in Chroma DB, similarity score should be extremely low or negative
    results = retrieve("unrelated random text about bananas and monkeys", score_floor=0.55)
    
    # Should be filtered out
    assert len(results) == 0


def test_max_retries_and_query_refinement(setup_e2e):
    """Test 7: Verify that grounding failures trigger query refinement up to MAX_RETRIES (2)."""
    user_id = "user-admin"
    query = "Draft a docx memo about Section 183."
    session_id = str(uuid.uuid4())
    
    # Track the queries retrieved
    retrieved_queries = []
    
    def custom_retrieve(query, session_id=None, score_floor=0.55, user_id=None):
        retrieved_queries.append(query)
        # Return empty list to force grounding failure and query refinement
        return []
        
    def custom_refine(original_q, failed_claim):
        return f"{original_q} refined with {failed_claim}"
        
    def custom_draft(q):
        # Return a single claim so that we do exactly 3 retrieves total
        return ["Claim 1 regarding Section 183"]

    with patch("src.orchestrator.main.retrieve", side_effect=custom_retrieve), \
         patch("src.orchestrator.main.mock_llm_refine_query", side_effect=custom_refine), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft):
         
        process_legal_request(query, user_id, session_id=session_id)
        
        # Max retries is 2. So we try:
        # Iteration 0: original query
        # Iteration 1: refined query 1
        # Iteration 2: refined query 2
        # Then fallback is triggered
        assert len(retrieved_queries) == 3
        assert retrieved_queries[0] == "Draft a docx memo about Section 183."
        assert "refined with" in retrieved_queries[1]
        assert "refined with" in retrieved_queries[2]


def test_browser_fallback_trigger(setup_e2e):
    """Test 8: Verify browser fallback is triggered if RAG grounding fails after max retries."""
    user_id = "user-admin"
    query = "Draft a docx memo about Section 999." # Not in corpus
    session_id = str(uuid.uuid4())
    
    def custom_parse(req):
        return {"output_format": "docx", "legal_question": "Section 999"}
        
    def custom_draft(q):
        return ["Section 999 is verified"]
        
    # Force retrieve to return empty list (RAG fails)
    with patch("src.orchestrator.main.retrieve", return_value=[]), \
         patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft), \
         patch("src.orchestrator.main.call_browser_fallback", return_value="Fallback Data Found") as mock_fallback:
         
        process_legal_request(query, user_id, session_id=session_id)
        
        # Verify fallback was called
        mock_fallback.assert_called_once_with("Section 999 is verified")
        
        # Verify grounding logs show browser fallback triggered for this session
        log_file = os.path.join(setup_e2e, "grounding_logs.json")
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            
        session_logs = [log for log in logs if log["session_id"] == session_id]
        steps = [log["step"] for log in session_logs]
        assert "browser_fallback_triggered" in steps


def test_multiple_claims_grounded(setup_e2e):
    """Test 9: Verify that multiple drafted claims are grounded individually."""
    user_id = "user-admin"
    query = "Draft a docx memo about Section 183 speeding and Section 185 drunk driving."
    session_id = str(uuid.uuid4())
    
    def custom_parse(req):
        return {"output_format": "docx", "legal_question": "Section 183 and 185"}
        
    def custom_draft(q):
        return [
            "Section 183 details speeding penalties",
            "Section 185 details drunk driving penalties"
        ]

    with patch("src.orchestrator.main.mock_llm_parse_request", side_effect=custom_parse), \
         patch("src.orchestrator.main.mock_llm_draft_claims", side_effect=custom_draft):
         
        process_legal_request(query, user_id, session_id=session_id)
        
        # Check logs for grounding checks on both claims
        log_file = os.path.join(setup_e2e, "grounding_logs.json")
        with open(log_file, "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f if line.strip()]
            
        session_logs = [log for log in logs if log["session_id"] == session_id]
        grounding_logs = [log for log in session_logs if log["step"] == "grounding_check"]
        
        # Grounding check must run at least once for each claim
        assert len(grounding_logs) >= 2
        
        claims_checked = [g["details"]["claim"] for g in grounding_logs]
        assert "Section 183 details speeding penalties" in claims_checked
        assert "Section 185 details drunk driving penalties" in claims_checked
