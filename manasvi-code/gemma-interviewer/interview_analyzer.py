import ollama
import config
import prompts
from data_models import InterviewDataRow
from datetime import datetime
import uuid

def calculate_vocal_metrics(text, duration):
    word_count = len(text.split())
    wpm = (word_count / duration) * 60 if duration > 0 else 0
    # Filler count logic is now removed.
    return {"wpm": round(wpm)}

def analyze_content_with_gemma(question, answer, model_name):
    print(f"Analyzing answer for question: '{question}'")
    prompt = prompts.CONTENT_ANALYSIS_PROMPT.format(question=question, answer=answer)
    messages = [{'role': 'user', 'content': prompt}]
    try:
        response = ollama.chat(model=model_name, messages=messages)
        analysis = {}
        for line in response['message']['content'].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                clean_key = key.strip().lower().replace(" ", "_").strip('\'"')
                analysis[clean_key] = value.strip()
        return analysis
    except Exception as e:
        print(f"Error during content analysis: {e}")
        return {}

def run_full_analysis(conversation_history, model_name, interview_type):
    print("\n--- Starting Post-Interview Analysis ---")
    validated_rows = []
    questions = [msg['content'] for msg in conversation_history if msg['role'] == 'assistant']
    answers = [{'text': msg['content'], 'duration': msg['duration']} for msg in conversation_history if msg['role'] == 'user']
    
    interview_id = uuid.uuid4()
    timestamp = datetime.now()

    for i, answer_data in enumerate(answers):
        if i >= len(questions): break
        
        question = questions[i]
        answer_text = answer_data['text']
        answer_duration = answer_data['duration']
        
        vocal_metrics = calculate_vocal_metrics(answer_text, answer_duration)
        content_analysis = analyze_content_with_gemma(question, answer_text, model_name)
        
        full_data = {
            "interview_id": interview_id, "timestamp": timestamp, "interview_type": interview_type,
            "question_number": i + 1, "question_text": question, "answer_text": answer_text,
            **vocal_metrics, **content_analysis
        }
        
        try:
            validated_row = InterviewDataRow(**full_data)
            validated_rows.append(validated_row)
        except Exception as e:
            print(f"--- Data Validation Error for question {i+1}: {e} ---")
            continue
            
    return validated_rows