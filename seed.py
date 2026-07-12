import os
import sqlite3
from src.rag import ingest_law_corpus, retrieve

def main():
    # 1. Initialize & Seed SQL DB
    print("Initializing and seeding permissions.db...")
    with open('src/sql/schema.sql', 'r') as f:
        schema = f.read()

    conn = sqlite3.connect('permissions.db')
    conn.executescript(schema)
    cursor = conn.cursor()

    # Clear any existing rows to ensure clean seeding
    cursor.execute("DELETE FROM role_permissions")
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM roles")
    cursor.execute("DELETE FROM permissions")

    # Seed mock tables (Using IDs consistent with tests)
    cursor.execute("INSERT INTO roles (id, name) VALUES ('role-admin', 'admin')")
    cursor.execute("INSERT INTO roles (id, name) VALUES ('role-viewer', 'viewer')")
    
    cursor.execute("INSERT INTO permissions (id, name) VALUES ('perm-generate', 'generate_legal_doc')")
    
    cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ('role-admin', 'perm-generate')")
    
    # Insert required test users
    cursor.execute("INSERT INTO users (id, name, role_id) VALUES ('u1', 'Test User', 'role-admin')")
    cursor.execute("INSERT INTO users (id, name, role_id) VALUES ('user-admin', 'Admin User', 'role-admin')")
    cursor.execute("INSERT INTO users (id, name, role_id) VALUES ('user-denied', 'Denied User', 'role-viewer')")

    conn.commit()
    conn.close()
    print("SQL Database successfully initialized!")

    # 2. Ingest Law Corpus (BNS 2023)
    # Re-ingestion logic is kept for full environment setup, but not strictly needed on every run.
    print("\nIngesting BNS 2023 PDF into Chroma...")
    ingest_law_corpus("data/law_corpus")
    print("Ingestion complete!")

    # 3. Test Retrieve Query
    print("\nRunning a test retrieval query:")
    results = retrieve("punishment for theft", top_k=2, score_floor=0.45, user_id="user-admin")
    if isinstance(results, dict) and "error" in results:
        print(f"Retrieval failed. Error details: {results}")
    else:
        for i, r in enumerate(results):
            print(f"[{i+1}] Score: {r['score']:.2%}, Source: '{r['source']}'")
            print(f"    Text: {r['text'][:200].strip()}...\n")

if __name__ == "__main__":
    main()
