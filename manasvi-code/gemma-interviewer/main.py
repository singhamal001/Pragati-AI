import sys
import torch
from transformers import pipeline
from piper.voice import PiperVoice
import speech_recognition as sr

# Import our custom modules
import config
from audio_processing import speak, listen
from gemma_logic import get_gemma_interviewer_response

def initialize_models():
    """Loads and initializes all the necessary models."""
    print("--- Initializing Models ---")
    try:
        print("Loading Whisper model...")
        transcriber = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL_NAME, device=config.DEVICE, return_timestamps=True)
        
        print("Loading Piper TTS model...")
        piper_voice = PiperVoice.load(config.PIPER_MODEL_PATH)
        
        print("Initializing SpeechRecognition...")
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = config.PAUSE_THRESHOLD
        microphone = sr.Microphone(sample_rate=config.MIC_SAMPLE_RATE)
        
        print("--- All models initialized successfully! ---")
        return transcriber, piper_voice, recognizer, microphone
    except Exception as e:
        print(f"FATAL ERROR during model initialization: {e}")
        sys.exit()

def run_interview(transcriber, piper_voice, recognizer, microphone):
    """Orchestrates the main interview flow."""
    conversation_history = []
    question_counter = 0

    # Stage 1: The Introduction
    print("\n--- The interview is about to begin. ---")
    intro_message = get_gemma_interviewer_response(conversation_history, config.GEMMA_MODEL_NAME)
    conversation_history.append({'role': 'assistant', 'content': intro_message})
    speak(intro_message, piper_voice)

    # Stage 2: The Question Loop
    while question_counter < config.MAX_QUESTIONS:
        print(f"\n--- Question {question_counter + 1} of {config.MAX_QUESTIONS} ---")
        
        user_answer = listen(recognizer, microphone, transcriber)
        if not user_answer:
            continue # If listening failed, loop again to re-listen

        conversation_history.append({'role': 'user', 'content': user_answer})
        question_counter += 1

        gemma_question = get_gemma_interviewer_response(conversation_history, config.GEMMA_MODEL_NAME)
        conversation_history.append({'role': 'assistant', 'content': gemma_question})
        
        # The last message from Gemma will be the conclusion, so we don't speak it here.
        # The loop will end, and we'll speak it in the conclusion stage.
        if question_counter < config.MAX_QUESTIONS:
            speak(gemma_question, piper_voice)

    # Stage 3: The Conclusion
    final_message = conversation_history[-1]['content']
    print("\n--- Interview Concluding ---")
    speak(final_message, piper_voice)
    
    print("\nInterview finished. Application will now close.")

if __name__ == "__main__":
    transcriber_model, piper_voice_model, recognizer_obj, mic_obj = initialize_models()
    run_interview(transcriber_model, piper_voice_model, recognizer_obj, mic_obj)
    sys.exit()