import os
import sys
import shutil
import subprocess
import re

def main():
    # Ensure stdout handles UTF-8 (emojis etc.) on Windows
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
            
    # 1. Clear the results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    if os.path.exists(results_dir):
        # We clean existing docx, xlsx, json files to ensure fresh run
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(results_dir, exist_ok=True)

    # 2. Run pytest
    print("Running E2E tests...")
    test_file = os.path.join(current_dir, "test_orchestrator.py")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", test_file],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # 3. Parse pytest output to construct the table
    # Example line: tests/test_orchestrator.py::test_permission_denied PASSED
    lines = result.stdout.split("\n")
    test_results = []
    
    pattern = re.compile(r"tests/test_orchestrator.py::(\w+)\s+(PASSED|FAILED|ERROR)")
    for line in lines:
        match = pattern.search(line)
        if match:
            test_name = match.group(1)
            status = match.group(2)
            
            # Map status to symbols/words
            if status == "PASSED":
                status_str = "✅ PASS"
            else:
                status_str = "❌ FAIL"
                
            # Documented description of what the test case covers
            descriptions = {
                "test_permission_denied": "Verify immediately rejected when user has no permission",
                "test_single_clean_offense_docx": "Verify single clean offense query generating docx memo",
                "test_multi_offense_comparison_xlsx": "Verify multi-offense comparison query generating xlsx memo",
                "test_ambiguous_query_clarification": "Verify ambiguous query triggers clarifying question",
                "test_no_matching_law_refusal": "Verify query with no matching law in corpus is refused",
                "test_rag_score_filtering": "Verify RAG filters out search hits below score floor (0.55)",
                "test_max_retries_and_query_refinement": "Verify query refined and retried up to MAX_RETRIES (2)",
                "test_browser_fallback_trigger": "Verify fallback search triggered when RAG fails",
                "test_multiple_claims_grounded": "Verify multiple claims are drafted and grounded individually"
            }
            
            desc = descriptions.get(test_name, "E2E Test case")
            test_results.append((test_name, desc, status_str))

    # 4. Generate the markdown table
    table_header = "| Test Case | Description | Status |\n|---|---|---|\n"
    table_rows = []
    for test_name, desc, status_str in test_results:
        table_rows.append(f"| `{test_name}` | {desc} | {status_str} |")
        
    summary_table = table_header + "\n".join(table_rows) + "\n"
    
    # 5. Save the summary table to tests/results/summary_table.md
    summary_file_path = os.path.join(results_dir, "summary_table.md")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("# E2E Test Execution Summary\n\n")
        f.write(summary_table)
        
    print("\n" + "="*40)
    print(" E2E TEST SUMMARY")
    print("="*40)
    print(summary_table)
    print(f"Results and summary table saved under {results_dir}")
    print("="*40)

if __name__ == '__main__':
    main()
