import os
import sys
import pytest
import tempfile

# Ensure workspace root is in sys.path so src.sql can be imported when running from src/sql
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.sql.auth import check_permission, get_connection

@pytest.fixture(autouse=True)
def setup_test_db():
    # Create a temporary file for the SQLite database
    db_fd, db_path = tempfile.mkstemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    
    # Read the schema file and run the migrations
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
        
    conn = get_connection()
    try:
        conn.executescript(schema_sql)
        
        # Seed test data
        cursor = conn.cursor()
        
        # Seed roles
        cursor.execute("INSERT INTO roles (id, name) VALUES (?, ?)", ("role-admin", "admin"))
        cursor.execute("INSERT INTO roles (id, name) VALUES (?, ?)", ("role-editor", "editor"))
        cursor.execute("INSERT INTO roles (id, name) VALUES (?, ?)", ("role-viewer", "viewer"))
        
        # Seed permissions
        cursor.execute("INSERT INTO permissions (id, name) VALUES (?, ?)", ("perm-read", "read"))
        cursor.execute("INSERT INTO permissions (id, name) VALUES (?, ?)", ("perm-write", "write"))
        cursor.execute("INSERT INTO permissions (id, name) VALUES (?, ?)", ("perm-delete", "delete"))
        
        # Seed role_permissions
        # Admin gets read, write, delete
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-admin", "perm-read"))
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-admin", "perm-write"))
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-admin", "perm-delete"))
        # Editor gets read, write
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-editor", "perm-read"))
        cursor.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES (?, ?)", ("role-editor", "perm-write"))
        # Viewer gets nothing
        
        # Seed users
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-admin", "Alice Admin", "role-admin"))
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-editor", "Bob Editor", "role-editor"))
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-viewer", "Charlie Viewer", "role-viewer"))
        cursor.execute("INSERT INTO users (id, name, role_id) VALUES (?, ?, ?)", ("user-norole", "Dave NoRole", None))
        
        conn.commit()
    finally:
        conn.close()
        
    yield
    
    # Cleanup
    os.close(db_fd)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]

def test_allowed_action():
    # Admin is allowed to delete
    assert check_permission("user-admin", "delete") is True
    # Editor is allowed to write
    assert check_permission("user-editor", "write") is True
    # Editor is allowed to read
    assert check_permission("user-editor", "read") is True

def test_denied_action():
    # Editor is denied to delete
    assert check_permission("user-editor", "delete") is False
    # Viewer is denied to write
    assert check_permission("user-viewer", "write") is False
    # User with no role is denied to read
    assert check_permission("user-norole", "read") is False

def test_unknown_user():
    # Non-existent user should be denied any action
    assert check_permission("user-nonexistent", "read") is False

def test_unknown_action():
    # Admin is denied an unknown action
    assert check_permission("user-admin", "unknown-action") is False
