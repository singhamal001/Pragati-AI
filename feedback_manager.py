# feedback_manager.py

import sqlite3
from data_models import InterviewDataRow

DB_FILE = "profiles.db"

def save_feedback_to_db(user_id: int, pydantic_rows: list[InterviewDataRow]):
    """
    Saves a list of validated feedback data rows to the SQLite database.
    """
    if not pydantic_rows:
        print("No validated feedback rows to save.")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        rows_to_insert = []
        for row_model in pydantic_rows:
            data = row_model.model_dump(mode='json')
            data['user_id'] = user_id
            rows_to_insert.append(data)
        
        cursor.executemany("""
            INSERT INTO feedback_reports (
                user_id, interview_id, timestamp, interview_type, question_number,
                question_text, answer_text, wpm, star_score, star_reason,
                keywords_score, keywords_reason, professionalism_score, professionalism_reason
            ) VALUES (
                :user_id, :interview_id, :timestamp, :interview_type, :question_number,
                :question_text, :answer_text, :wpm, :star_score, :star_reason,
                :keywords_score, :keywords_reason, :professionalism_score, :professionalism_reason
            )
        """, rows_to_insert)

        conn.commit()
        print(f"Report saved to database for user_id: {user_id} with Interview ID: {pydantic_rows[0].interview_id}")

    except sqlite3.Error as e:
        print(f"Database error saving feedback report: {e}")
    finally:
        if conn:
            conn.close()

def get_all_interviews_for_user(user_id: int):
    """
    Fetches a summary of all past interview sessions for a specific user.
    Returns a list of dictionaries, each containing interview_id, type, and timestamp.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT interview_id, interview_type, timestamp
            FROM feedback_reports
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        
        interviews = [dict(row) for row in cursor.fetchall()]
        return interviews

    except sqlite3.Error as e:
        print(f"Database error fetching interview list: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_report_details_by_interview_id(interview_id: str):
    """
    Fetches all the feedback details (all question/answer rows) for a
    single, specific interview session.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM feedback_reports
            WHERE interview_id = ?
            ORDER BY question_number ASC
        """, (interview_id,))
        
        report_details = [dict(row) for row in cursor.fetchall()]
        return report_details

    except sqlite3.Error as e:
        print(f"Database error fetching report details: {e}")
        return []
    finally:
        if conn:
            conn.close()
