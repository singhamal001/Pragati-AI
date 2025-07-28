import pandas as pd
from gemma_logic import get_simple_response
import re

def parse_nth(text):
    """Helper to find numbers like '3rd', '5', 'fifth' in a string."""
    text = text.lower()
    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[0])
    
    word_map = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": 1, "latest": 1}
    for word, num in word_map.items():
        if word in text:
            return num
    return None

def handle_feedback_request(user_query, model_name):
    try:
        df = pd.read_csv('feedback_reports.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        return "It looks like you don't have any saved feedback reports yet."

    query = user_query.lower()
    
    # --- NEW: More robust keyword matching ---
    interview_type = None
    if any(word in query for word in ["hr", "salary", "negotiation"]):
        interview_type = "Salary Negotiation"
    elif any(word in query for word in ["background", "behavioral", "project"]):
        interview_type = "Background"
    
    # If the type is still unknown, we can't proceed
    if not interview_type:
        return "I can provide feedback on 'Background' or 'Salary Negotiation' interviews. Please specify which type you're interested in."

    # Route to the correct function based on keywords
    if "compare" in query:
        num_reports = parse_nth(query) or 3 # Default to 3 if not specified
        return get_comparison_report(df, model_name, interview_type, num_reports)
    
    elif "last" in query or "latest" in query or "recent" in query:
        n = parse_nth(query) or 1
        return get_nth_last_interview_report(df, model_name, interview_type, n)
        
    else:
        # If the query is specific but doesn't fit a pattern, ask for clarification
        return f"I understood you're asking about {interview_type} interviews. Could you clarify if you want the 'latest' report or a 'comparison'?"

def get_nth_last_interview_report(df, model_name, interview_type, n):
    """Gets the Nth last report of a specific type from the provided DataFrame."""
    print(f"Fetching {n}-th last '{interview_type}' interview...")
    
    type_df = df[df['interview_type'] == interview_type]
    if type_df.empty: return f"I couldn't find any saved '{interview_type}' interviews."
        
    unique_interviews = type_df.sort_values('timestamp', ascending=False)['interview_id'].unique()
    
    if n > len(unique_interviews): return f"You only have {len(unique_interviews)} saved '{interview_type}' interviews. I can't find the {n}-th last one."
        
    target_id = unique_interviews[n - 1]
    report_data = df[df['interview_id'] == target_id]
    
    prompt = f"Summarize the following interview data for the user. This was their {n}-th last {interview_type} interview. Data:\n{report_data.to_string()}"
    return get_simple_response(prompt, model_name)

def get_comparison_report(df, model_name, interview_type, num_reports):
    """Compares the last N reports of a specific type from the provided DataFrame."""
    print(f"Comparing last {num_reports} '{interview_type}' interviews...")

    type_df = df[df['interview_type'] == interview_type]
    if type_df.empty: return f"I couldn't find any saved '{interview_type}' interviews to compare."

    recent_interviews = type_df.sort_values('timestamp', ascending=False)['interview_id'].unique()[:num_reports]
    report_data = df[df['interview_id'].isin(recent_interviews)]

    prompt = f"Analyze the user's performance across their last {len(recent_interviews)} '{interview_type}' interviews. Identify trends and improvements. Data:\n{report_data.to_string()}"
    return get_simple_response(prompt, model_name)