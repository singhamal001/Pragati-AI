import sys
import random
from transformers import pipeline
from piper.voice import PiperVoice
import speech_recognition as sr
import numpy as np
import pandas as pd

# Import all custom modules
import config
import prompts
from audio_processing import speak, listen
from gemma_logic import get_orchestrator_decision, get_interview_response, generate_final_feedback_report, get_simple_response, get_feedback_decision
from interview_analyzer import run_full_analysis
from data_storage import save_report_to_csv
from report_generator import get_nth_last_interview_report, get_comparison_report

def initialize_models():
    """Loads and initializes all the necessary models. Exits on failure."""
    print("--- Initializing Models ---")
    try:
        transcriber = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL_NAME, device=config.DEVICE, return_timestamps=True)
        piper_voice = PiperVoice.load(config.PIPER_MODEL_PATH)
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = config.PAUSE_THRESHOLD
        microphone = sr.Microphone(sample_rate=config.MIC_SAMPLE_RATE)
        print("--- All models initialized successfully! ---")
        return transcriber, piper_voice, recognizer, microphone
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        sys.exit()

def run_interview(transcriber, piper_voice, recognizer, microphone, interview_type):
    """A robust interview function that relies on the stateful LLM prompt to guide the conversation."""
    conversation_history = []
    prompt_template = prompts.SALARY_NEGOTIATION_PROMPT if interview_type == "Salary Negotiation" else prompts.BACKGROUND_INTERVIEW_PROMPT
    print(f"--- Starting {interview_type} interview ---")

    for _ in range(15): # Safety net of 15 turns
        gemma_response = get_interview_response(conversation_history, config.GEMMA_MODEL_NAME, prompt_template)
        conversation_history.append({'role': 'assistant', 'content': gemma_response})
        speak(gemma_response, piper_voice)

        conclusion_phrases = ["thank you for your time", "we'll be in touch", "end the simulation", "conclude our discussion"]
        if any(phrase in gemma_response.lower() for phrase in conclusion_phrases):
            print("--- Interview concluded naturally by Gemma ---")
            break

        user_answer, duration = listen(recognizer, microphone, transcriber, piper_voice)
        if not user_answer:
            speak("I didn't quite catch that, let's try that again.", piper_voice)
            conversation_history.pop()
            continue
        
        conversation_history.append({'role': 'user', 'content': user_answer, 'duration': duration})
    else:
        print("--- Ending interview: Maximum turns reached ---")
        speak("We've covered a lot of ground, so let's wrap up there. Thank you for your time.", piper_voice)

    print(f"\n--- {interview_type} Interview Finished. Generating feedback... ---")
    analysis_results = run_full_analysis(conversation_history, config.GEMMA_MODEL_NAME, interview_type)
    save_report_to_csv(analysis_results)
    
    if analysis_results:
        analysis_summary = ""
        for item in analysis_results:
            analysis_summary += (f"For the question about '{item.question_text[:40]}...', your STAR score was {item.star_score} because '{item.star_reason}'. Your Professionalism score was {item.professionalism_score} because '{item.professionalism_reason}'.\n")
    else:
        analysis_summary = "No analysis data was generated for this interview."
    
    speak("Your interview data has been saved. Would you like to hear your feedback report now?", piper_voice)
    # Use the fast, simple confirmation for this predictable question
    if listen_for_simple_confirmation(recognizer, microphone, transcriber, piper_voice):
        final_spoken_report = generate_final_feedback_report(analysis_summary, config.GEMMA_MODEL_NAME)
        speak(final_spoken_report, piper_voice)
    else:
        speak("Okay. You can ask for a feedback report from the main menu later.", piper_voice)

def handle_feedback_mode(recognizer, microphone, transcriber, piper_voice):
    """Handles the feedback interaction with a clear, guided menu."""
    menu_text = """
    Of course. I can provide several types of reports. Please say the option you would like:
    1. A detailed summary of your most recent Salary negotiation interview.
    2. A comparison of your last 3 Background interviews.
    3. A comparison of your last 3 Salary Negotiation sessions.
    4. A report on your 3rd last Background interview.
    """
    speak(menu_text, piper_voice)
    
    # Give the user up to 3 chances to select a valid option
    for _ in range(3):
        choice_text, _ = listen(recognizer, microphone, transcriber, piper_voice)
        if not choice_text:
            speak("I'm sorry, I didn't get that. Please say the number of the option you'd like.", piper_voice)
            continue

        # Use the intelligent router to understand the choice
        decision = get_feedback_decision(choice_text, config.GEMMA_MODEL_NAME)
        
        if decision:
            speak(f"Understood. Analyzing your records. Please wait.", piper_voice)
            
            # Load the dataframe once, only when needed
            try:
                df = pd.read_csv('feedback_reports.csv')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except FileNotFoundError:
                speak("It looks like you don't have any saved feedback reports yet.", piper_voice)
                return

            if decision['function'] == 'get_nth_last_report':
                report = get_nth_last_interview_report(df, config.GEMMA_MODEL_NAME, decision['type'], decision['n'])
            elif decision['function'] == 'get_comparison_report':
                report = get_comparison_report(df, config.GEMMA_MODEL_NAME, decision['type'], decision['n'])
            else:
                report = "I understood the command, but there was an error executing it."
            
            speak(report, piper_voice)
            return # Exit successfully

        else:
            speak("I'm sorry, I didn't understand that option.", piper_voice)

    speak("I'm having trouble understanding. Let's return to the main menu.", piper_voice)

def listen_for_simple_confirmation(recognizer, microphone, transcriber, piper_voice):
    """A fast, simple, keyword-based function to listen for a 'yes' or 'no' response."""
    print("Waiting for simple confirmation (yes/no)...")
    response, _ = listen(recognizer, microphone, transcriber, piper_voice)
    if response and "yes" in response.lower():
        return True
    return False

def listen_for_confirmation_llm(recognizer, microphone, transcriber, piper_voice):
    """Uses the LLM to intelligently determine if a response is a 'yes' or 'no'."""
    print("Waiting for LLM confirmation...")
    response_text, _ = listen(recognizer, microphone, transcriber, piper_voice)
    if not response_text:
        return None

    prompt = prompts.CONFIRMATION_PROMPT.format(user_response=response_text)
    decision = get_simple_response(prompt, config.GEMMA_MODEL_NAME)

    if "YES" in decision.upper():
        return True
    if "NO" in decision.upper():
        return False
    return None

def main():
    """The main application orchestrator."""
    transcriber, piper_voice, recognizer, microphone = initialize_models()
    speak("Hello! I'm Gemma, your interview coach. Please choose: Background Interview, Salary Negotiation Interview or Feedback Report.", piper_voice)

    while True:
        print("\n--- Waiting for your command ---")
        command, _ = listen(recognizer, microphone, transcriber, piper_voice)
        if not command: continue

        intent = get_orchestrator_decision(command, config.GEMMA_MODEL_NAME)
        
        confirmed = False
        if intent in ['BACKGROUND_INTERVIEW', 'SALARY_NEGOTIATION', 'FEEDBACK']:
            confirmation_prompt = f"I understood that you want to start a {intent.replace('_', ' ').title()}. Is that correct?"
            speak(confirmation_prompt, piper_voice)
            # Use the intelligent, flexible LLM confirmation for the main menu
            confirmation_result = listen_for_confirmation_llm(recognizer, microphone, transcriber, piper_voice)
            
            if confirmation_result is True:
                confirmed = True
            elif confirmation_result is False:
                speak("My mistake. Let's try again. What would you like to do?", piper_voice)
            else:
                speak("I didn't quite understand your response. Let's try again from the main menu.", piper_voice)
        elif intent == 'EXIT':
            confirmed = True

        if not confirmed:
            continue

        if intent == 'BACKGROUND_INTERVIEW':
            run_interview(transcriber, piper_voice, recognizer, microphone, "Background")
        elif intent == 'SALARY_NEGOTIATION':
            run_interview(transcriber, piper_voice, recognizer, microphone, "Salary Negotiation")
        elif intent == 'FEEDBACK':
            handle_feedback_mode(recognizer, microphone, transcriber, piper_voice)
        elif intent == 'EXIT':
            speak("Goodbye!", piper_voice)
            sys.exit()
        else:
            speak("I'm sorry, I didn't understand that command.", piper_voice)
        
        speak("I'm ready for your next command.", piper_voice)

if __name__ == "__main__":
    main()