import os
import sqlite3
from urllib.parse import urlparse

def get_connection():
    """
    Returns a database connection based on the DATABASE_URL environment variable.
    Defaults to a SQLite connection to 'permissions.db' if DATABASE_URL is not set.
    """
    db_url = os.environ.get("DATABASE_URL")
    
    if db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
        import psycopg2
        return psycopg2.connect(db_url)
    
    # Fallback to SQLite
    sqlite_url = db_url if db_url else "sqlite:///permissions.db"
    
    if sqlite_url.startswith("sqlite:///"):
        db_path = sqlite_url[10:]
    else:
        db_path = sqlite_url
        
    conn = sqlite3.connect(db_path)
    # Enable foreign keys for SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def check_permission(user_id: str, action: str) -> bool:
    """
    Checks if a user has the permission to perform a given action.
    Joins across users, roles, permissions, and role_permissions tables.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    is_postgres = db_url.startswith("postgres://") or db_url.startswith("postgresql://")
    
    placeholder = "%s" if is_postgres else "?"
    
    query = f"""
        SELECT COUNT(*)
        FROM users u
        JOIN roles r ON u.role_id = r.id
        JOIN role_permissions rp ON r.id = rp.role_id
        JOIN permissions p ON rp.permission_id = p.id
        WHERE u.id = {placeholder} AND p.name = {placeholder}
    """
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, (user_id, action))
        row = cursor.fetchone()
        if row:
            return int(row[0]) > 0
        return False
    except Exception:
        # Re-raise or return False depending on how robust we want to be.
        # Re-raising is generally better for debugging db/schema issues.
        raise
    finally:
        conn.close()
