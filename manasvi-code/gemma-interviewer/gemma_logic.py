import ollama

def get_gemma_interviewer_response(conversation_history, model_name):
    """
    Sends the entire conversation history to Gemma to get the next response.
    """
    print(">> Gemma is thinking...")
    
    instructional_prompt = f"""
Instruction: You are an expert interviewer named Gemma. Your goal is to conduct a friendly but thorough interview. Your persona is encouraging and professional.
Do not be rude but you can be critical if you feel confused with an answer, or if you want them to elaborate or the answer did not make sense with what you asked.

RULES:
1.  Ask only one question at a time.
2.  Your questions must be dynamic and based on the user's previous answers. If they mention a technology, ask a follow-up about it.
3.  Do not use any special characters like '*' or '**' in your answer. Only use plain text.
4.  The interview will proceed in stages. Your current stage is determined by the conversation history.

CONVERSATION FLOW:
-   **Stage 1 (Introduction):** If the conversation is empty, introduce yourself and ask the user to tell you about themselves to make them comfortable.
-   **Stage 2 (Interview Questions):** After their introduction, ask your first technical question. Continue asking follow-up questions based on their answers.
-   **Stage 3 (Conclusion):** After about 5-7 questions, conclude the interview gracefully. Thank the user for their time and end on a positive note. Do not ask a question in the conclusion. Summarize the conversation as an interviewer in the conclusion.

Here is the conversation so far:
{format_history_for_prompt(conversation_history)}

Based on the rules and the history, provide your next single response or question.
"""
    
    messages_for_api = [{'role': 'user', 'content': instructional_prompt}]
    
    try:
        response = ollama.chat(model=model_name, messages=messages_for_api)
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

def format_history_for_prompt(history):
    """Helper function to format the conversation history for the prompt."""
    if not history:
        return "The conversation has not started yet."
    
    formatted_string = ""
    for message in history:
        role = "You (The Candidate)" if message['role'] == 'user' else "Gemma (The Interviewer)"
        formatted_string += f"{role}:\n{message['content']}\n\n"
    return formatted_string