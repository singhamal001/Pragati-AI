import prompts
from data_models import InterviewDataRow
from datetime import datetime
import uuid

def calculate_vocal_metrics(text, duration):
    word_count = len(text.split())
    wpm = (word_count / duration) * 60 if duration > 0 else 0
    return {"wpm": round(wpm)}

def analyze_content_with_gemma(gemma_model, process_func, question, answer):
    """MODIFIED to accept the gemma_model instance and processing function."""
    print(f"Analyzing answer for question: '{question}'")
    prompt = prompts.CONTENT_ANALYSIS_PROMPT.format(question=question, answer=answer)
    full_prompt = f"[INST]\n{prompt}\n[/INST]"
    
    try:
        response_text = process_func(full_prompt, max_tokens=300)
        analysis = {}
        for line in response_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                clean_key = key.strip().lower().replace(" ", "_").strip('\'"')
                analysis[clean_key] = value.strip()
        return analysis
    except Exception as e:
        print(f"Error during content analysis: {e}")
        return {}

def run_full_analysis(gemma_model, process_func, conversation_history, interview_type):
    """MODIFIED to pass the model and process_func down."""
    print("\n--- Starting Post-Interview Analysis ---")
    validated_rows = []
    questions = [msg['content'] for msg in conversation_history if msg['role'] == 'assistant']
    answers = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
    
    interview_id = uuid.uuid4()
    timestamp = datetime.now()

    for i, answer_text in enumerate(answers):
        if i >= len(questions): break
        
        question = questions[i]
        
        answer_duration = (len(answer_text.split()) / 150) * 60

        vocal_metrics = calculate_vocal_metrics(answer_text, answer_duration)
        content_analysis = analyze_content_with_gemma(gemma_model, process_func, question, answer_text)
        
        full_data = {
            "interview_id": interview_id, "timestamp": timestamp, "interview_type": interview_type,
            "question_number": i + 1, "question_text": question, "answer_text": answer_text,
            "wpm": vocal_metrics['wpm'], **content_analysis
        }
        
        try:
            validated_row = InterviewDataRow(**full_data)
            validated_rows.append(validated_row)
        except Exception as e:
            print(f"--- Data Validation Error for question {i+1}: {e} ---")
            continue
            
    return validated_rows