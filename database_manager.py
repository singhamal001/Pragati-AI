# database_manager.py

import sqlite3
import json
from datetime import datetime

DB_FILE = "profiles.db"

def initialize_database():
    """Initializes the database and creates tables if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                age INTEGER NOT NULL, 
                preferences TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            print("No users found. Creating default Admin profile...")
            # MODIFIED: Default screen is now the interview screen
            admin_preferences = json.dumps({
                "last_screen": "interview_screen",
                "onboarding_complete": True 
            })
            cursor.execute(
                "INSERT INTO users (username, role, age, preferences) VALUES (?, ?, ?, ?)",
                ("Admin", "admin", 99, admin_preferences)
            )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()

def get_all_users():
    """Fetches all users from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, age FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        return users
    except sqlite3.Error as e:
        print(f"Database error fetching users: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_user_by_username(username):
    """Fetches a single user's complete data by their username."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        if user_data:
            return dict(user_data)
        return None
    except sqlite3.Error as e:
        print(f"Database error fetching user {username}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def add_user(name, age):
    """Adds a new user to the database with onboarding set to false."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # MODIFIED: Default screen is now the interview screen
        new_user_prefs = json.dumps({
            "last_screen": "interview_screen",
            "onboarding_complete": False,
            "profile_summary": {}
        })
        cursor.execute(
            "INSERT INTO users (username, role, age, preferences) VALUES (?, ?, ?, ?)",
            (name, "user", int(age), new_user_prefs)
        )
        conn.commit()
        return True, f"Success: User '{name}' added."
    except sqlite3.IntegrityError:
        return False, f"Error: Username '{name}' already exists."
    except sqlite3.Error as e:
        return False, f"Database Error: {e}"
    finally:
        if conn:
            conn.close()

def remove_user(user_id):
    """Removes a user and their entire conversation history."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversation_history WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True, "Success: User and their history removed."
    except sqlite3.Error as e:
        return False, f"Database Error: {e}"
    finally:
        if conn:
            conn.close()

def add_message_to_history(user_id, role, content):
    """Adds a single message to the conversation history table."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error adding message: {e}")
    finally:
        if conn:
            conn.close()

def get_conversation_history(user_id):
    """Retrieves and formats the entire conversation for a user."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM conversation_history WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,)
        )
        history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        return history
    except sqlite3.Error as e:
        print(f"Database error getting history: {e}")
        return []
    finally:
        if conn:
            conn.close()

def update_user_preferences(user_id, new_preferences):
    """Updates the preferences JSON for a specific user."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        preferences_json = json.dumps(new_preferences)
        cursor.execute(
            "UPDATE users SET preferences = ? WHERE id = ?",
            (preferences_json, user_id)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error updating preferences: {e}")
    finally:
        if conn:
            conn.close()