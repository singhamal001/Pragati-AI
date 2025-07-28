# interview_flow_manager.py
from gemma_logic import get_simple_response, format_history_for_prompt
import prompts

def get_topics_from_llm(conversation_history, model_name):
    """Uses the LLM to intelligently extract topics from the last user response."""
    last_user_answer = ""
    for msg in reversed(conversation_history):
        if msg['role'] == 'user':
            last_user_answer = msg['content']
            break
    if not last_user_answer: return set()

    prompt = prompts.TOPIC_EXTRACTION_PROMPT.format(last_user_answer=last_user_answer)
    try:
        response = get_simple_response(prompt, model_name)
        if "none" in response.lower(): return set()
        return set(topic.strip() for topic in response.split(','))
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return set()

def should_end_interview(conversation_history, interview_type, current_turn, min_turns, topics_covered):
    if current_turn < min_turns:
        return False, "Minimum turns not reached"
    if has_natural_conclusion_indicators(conversation_history):
        return True, "Natural conclusion detected"
    
    if interview_type == "HR & Salary":
        return should_end_hr_specific(conversation_history)
    else:
        return should_end_background_specific(conversation_history, topics_covered, current_turn)

def should_end_hr_specific(conversation_history):
    recent_messages = [msg['content'].lower() for msg in conversation_history[-4:]]
    indicators = ["final offer", "best offer", "budget constraints", "we'll be in touch"]
    for message in recent_messages:
        if any(indicator in message for indicator in indicators):
            return True, "Salary negotiation completed"
    return False, "Negotiation ongoing"

def should_end_background_specific(conversation_history, topics_covered, current_turn):
    expected_topics = {"project", "experience", "technical", "challenge", "team", "leadership"}
    coverage = len(topics_covered.intersection(expected_topics)) / len(expected_topics)
    
    if coverage >= 0.6 and current_turn >= 7:
        return True, "Sufficient topics covered"
    if is_conversation_stagnating(conversation_history):
        return True, "Conversation stagnating"
    if current_turn >= 10:
        return True, "Extensive discussion completed"
    return False, "More discussion needed"

def is_conversation_stagnating(conversation_history):
    if len(conversation_history) < 6: return False
    assistant_msgs = [msg['content'] for msg in conversation_history[-6:] if msg['role'] == 'assistant']
    if len(assistant_msgs) < 2: return False
    
    msg1_words = set(assistant_msgs[-2].lower().split())
    msg2_words = set(assistant_msgs[-1].lower().split())
    if not msg1_words or not msg2_words: return False
    
    if len(msg1_words.intersection(msg2_words)) / len(msg1_words) > 0.7:
        return True
    return False

def has_natural_conclusion_indicators(conversation_history):
    if len(conversation_history) < 2: return False
    recent_messages = [msg['content'].lower() for msg in conversation_history[-3:]]
    phrases = ["thank you for your time", "that covers everything", "no more questions", "we'll be in touch"]
    return any(any(phrase in message for phrase in phrases) for message in recent_messages)

def generate_interview_conclusion(conversation_history, interview_type, model_name):
    prompt = prompts.GENERATE_CONCLUSION_PROMPT.format(
        interview_type=interview_type,
        history=format_history_for_prompt(conversation_history)
    )
    return get_simple_response(prompt, model_name)