import speech_recognition as sr
import torch
from transformers import pipeline
import ollama
import pyttsx3

# --- Configuration ---
MODEL_NAME = "openai/whisper-base.en"

# --- TTS Engine Initialization ---
try:
    tts_engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    tts_engine = None

# --- Initialization of Whisper ---
print("Initializing the speech-to-text model...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=device
)
print("Speech-to-text model initialized.")

# --- TTS Function ---
# def speak(text):
#     """Converts text to speech."""
#     print(f"\n< Gemma: {text}")
#     if tts_engine:
#         tts_engine.say(text)
#         tts_engine.runAndWait()

# # --- Corrected Gemma Interaction Function ---
# def get_gemma_response(text):
#     """Sends text to the Ollama Gemma model with instructions injected into the user prompt."""
#     if not text:
#         return "I didn't catch that. Could you please repeat?"
        
#     print(f"\n> You said: {text}")
#     print(">> Gemma is thinking...")
    
#     # --- CORRECTED PROMPT INJECTION ---
#     # We create a single, detailed prompt string.
#     full_prompt = f"""
# Instruction: You are a helpful assistant for a visually impaired user. Your goal is to be clear and concise. Please keep all your answers under 100 words.

# ---

# User's Question: "{text}"
# """
    
#     try:
#         # We now send this single, combined prompt as the user's content.
#         response = ollama.chat(
#             model='gemma3n:e4b',
#             messages=[
#                 {
#                     'role': 'user',
#                     'content': full_prompt,
#                 }
#             ]
#         )
#         return response['message']['content']
#     except Exception as e:
#         return f"Error communicating with Ollama: {e}"

def speak(text):
    """
    Converts text to speech using a new, isolated TTS engine instance
    for each call to ensure reliability.
    """
    # First, print the response so the user sees it immediately.
    print(f"\n< Gemma: {text}")
    
    # --- Defensive Check ---
    # Don't try to speak if the text is empty or just whitespace.
    if not text or not text.strip():
        print("TTS Skipped: No text to speak.")
        return

    try:
        # --- Isolated Engine Pattern ---
        # Initialize a new engine instance every time.
        tts_engine = pyttsx3.init()
        
        # Say the text.
        tts_engine.say(text)
        
        # Block until speaking is complete.
        tts_engine.runAndWait()
        
        # The engine is automatically cleaned up when the function exits.
        
    except Exception as e:
        # If TTS fails for any reason, we print the error but don't crash.
        print(f"\n--- TTS Engine Error ---")
        print(f"An error occurred while trying to speak: {e}")
        print(f"This can sometimes happen due to audio driver conflicts.")
        print(f"The text response was: {text}")
        print(f"------------------------")

def get_gemma_response(text):
    if not text:
        return "I didn't catch that. Could you please repeat?"
    print(f"\n> You said: {text}")
    print(">> Gemma is thinking...")
    full_prompt = f"""
Instruction: You are a helpful assistant for a visually impaired user. Your goal is to be clear and concise. Please keep all your answers under 100 words.
---
User's Question: "{text}"
"""
    try:
        response = ollama.chat(
            model='gemma3n:e4b',
            messages=[{'role': 'user', 'content': full_prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {e}"


def main():
    """Main loop to listen, transcribe, respond, and speak."""
    
    r = sr.Recognizer()
    r.pause_threshold = 4.0
    mic = sr.Microphone(sample_rate=16000)

    with mic as source:
        print("\nCalibrating microphone... Please be silent for a moment.")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Microphone calibrated.")

    speak("Hello, I am ready. How can I help you?")

    while True:
        print("\nListening for you to speak...")
        try:
            with mic as source:
                audio_data = r.listen(source)

            print("Processing audio...")
            wav_bytes = audio_data.get_wav_data()
            transcribed_text = transcriber(wav_bytes)["text"].strip()

            gemma_answer = get_gemma_response(transcribed_text)
            
            speak(gemma_answer)

        except sr.UnknownValueError:
            speak("I'm sorry, I couldn't understand the audio. Please try again.")
        except sr.RequestError as e:
            speak(f"There was a service error; {e}")
        except KeyboardInterrupt:
            speak("Goodbye!")
            break

if __name__ == "__main__":
    main()