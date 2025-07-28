# gemma_logic.py
import ollama
import prompts
import re

def get_orchestrator_decision(user_command, model_name):
    prompt = prompts.ORCHESTRATOR_PROMPT.format(user_command=user_command)
    return get_simple_response(prompt, model_name).strip().upper()

def get_interview_response(conversation_history, model_name, prompt_template):
    """
    Gets the next response for ANY ongoing interview using the specified prompt template.
    This single function now handles both HR and Background interviews.
    """
    print(">> Gemma is thinking...")
    
    # Format the chosen prompt with the current state of the interview
    prompt = prompt_template.format(
        history=format_history_for_prompt(conversation_history)
    )
    
    # We use get_simple_response to execute the call
    return get_simple_response(prompt, model_name)

def generate_final_feedback_report(analysis_summary, model_name):
    prompt = prompts.FINAL_SUMMARY_PROMPT.format(analysis_summary=analysis_summary)
    return get_simple_response(prompt, model_name)

def get_simple_response(prompt, model_name):
    messages = [{'role': 'user', 'content': prompt}]
    try:
        response = ollama.chat(model=model_name, messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"Error communicating with Gemma: {e}")
        return "I seem to be having trouble thinking right now. Please try again."

def format_history_for_prompt(history):
    if not history: return "The conversation has not started yet."
    formatted_string = ""
    for message in history:
        role = "You (The Candidate)" if message['role'] == 'user' else "Gemma (The Interviewer)"
        formatted_string += f"{role}:\n{message['content']}\n\n"
    return formatted_string

def get_feedback_decision(user_query, model_name):
    """Uses the LLM to parse a user's feedback request into a structured command."""
    prompt = prompts.FEEDBACK_ORCHESTRATOR_PROMPT.format(user_query=user_query)
    
    raw_response = get_simple_response(prompt, model_name)
    
    # --- Parse the structured response ---
    try:
        parts = [part.strip() for part in raw_response.split(',')]
        if len(parts) == 3:
            function_name = parts[0]
            interview_type = parts[1]
            n = int(parts[2])
            # Basic validation
            if function_name in ['get_nth_last_report', 'get_comparison_report'] and interview_type in ['Background', 'HR & Salary']:
                return {"function": function_name, "type": interview_type, "n": n}
    except (ValueError, IndexError) as e:
        print(f"Error parsing feedback decision: {e}. Raw response: '{raw_response}'")
    
    # Return None if parsing fails
    return None