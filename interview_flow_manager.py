# interview_flow_manager.py

from gemma_logic import format_history_for_prompt
import prompts

def get_topics_from_llm(gemma_model, process_func, conversation_history):
    """MODIFIED: Uses the passed-in model and process function."""
    last_user_answer = ""
    for msg in reversed(conversation_history):
        if msg['role'] == 'user':
            last_user_answer = msg['content']
            break
    if not last_user_answer: return set()

    prompt = prompts.TOPIC_EXTRACTION_PROMPT.format(last_user_answer=last_user_answer)
    full_prompt = f"[INST]\n{prompt}\n[/INST]"
    try:
        response = process_func(full_prompt, max_tokens=50)
        if "none" in response.lower(): return set()
        return set(topic.strip() for topic in response.split(','))
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return set()

def should_end_interview(conversation_history, interview_type, current_turn):
    """
    MODIFIED: Simplified to not need topics_covered directly,
    as stagnation is the more critical check for this problem.
    """
    if current_turn < 4:  # Ensure a minimum of 3 questions are asked
        return False, "Minimum turns not reached"
    
    if has_natural_conclusion_indicators(conversation_history):
        return True, "Natural conclusion detected in user response."

    if is_conversation_stagnating(conversation_history):
        return True, "Conversation is stagnating."

    if interview_type == "Background" and current_turn >= 10:
        return True, "Max turns reached for Background interview."
        
    return False, "More discussion needed"

def is_conversation_stagnating(conversation_history):
    """
    Checks if the last two AI responses are too similar. This is the key function
    to prevent the looping you are seeing.
    """
    if len(conversation_history) < 4: return False
    
    # Get the last two messages from the assistant
    assistant_msgs = [msg['content'] for msg in conversation_history if msg['role'] == 'assistant']
    if len(assistant_msgs) < 2: return False
    
    msg1_words = set(assistant_msgs[-2].lower().split())
    msg2_words = set(assistant_msgs[-1].lower().split())
    
    # Avoid division by zero if a message is empty
    if not msg1_words or not msg2_words: return False
    
    # Calculate Jaccard similarity
    intersection_len = len(msg1_words.intersection(msg2_words))
    union_len = len(msg1_words.union(msg2_words))
    
    if union_len == 0: return False

    similarity = intersection_len / union_len
    
    # If the similarity is very high (e.g., > 60%), we are stagnating.
    if similarity > 0.6:
        print(f"DEBUG: Stagnation detected! Similarity: {similarity:.2f}")
        return True
        
    return False

def has_natural_conclusion_indicators(conversation_history):
    if len(conversation_history) < 2: return False
    last_user_message = ""
    for msg in reversed(conversation_history):
        if msg['role'] == 'user':
            last_user_message = msg['content'].lower()
            break
    
    phrases = ["that's all", "that covers everything", "no more questions", "i'm done"]
    return any(phrase in last_user_message for phrase in phrases)