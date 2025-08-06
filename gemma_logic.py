# gemma_logic.py
import prompts
import re

def get_interview_response(gemma_model, process_func, current_session_log, prompt_template):
    """
    Gets the next response for an interview.
    """
    print(">> Gemma is thinking...")
    
    prompt = prompt_template.format(
        history=format_history_for_prompt(current_session_log)
    )
    
    full_prompt = f"[INST]\n{prompt}\n[/INST]"
    
    return process_func(full_prompt, max_tokens=250)

def format_history_for_prompt(history):
    """
    Formats a list of dictionaries into a string for the LLM prompt.
    """
    if not history: return "The conversation has not started yet."
    formatted_string = ""
    for message in history:
        role = "Candidate" if message['role'] == 'user' else "Interviewer (Gemma)"
        formatted_string += f"{role}:\n{message['content']}\n\n"
    return formatted_string