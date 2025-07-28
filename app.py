# app.py

import customtkinter as ctk
import threading
import speech_recognition as sr
import whisper
from llama_cpp import Llama
import os
from pathlib import Path
import json

from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

import database_manager as db
from ui_components import WelcomeFrame, AdminDashboard, MainAppFrame

# --- Constants ---
MODEL_PATH = "./model/gemma-3n-e2b-it.Q2_K_M.gguf"
PIPER_MODEL_PATH = "./model/en_US-hfc_female-medium.onnx"
BEEP_SOUND_PATH = "./assets/beep.wav"
ONBOARDING_AUDIO_FILES = [
    "./assets/instructions_part1.wav",
    "./assets/instructions_part2.wav",
]

# --- AI Personas ---
AI_PERSONAS = {
    "ONBOARDING_SPECIALIST": """
    You are Gemma, a friendly and empathetic Onboarding Specialist for a new, visually impaired user. Your mission is to conduct a short, welcoming interview to personalize their experience. The user has already received instructions.
    Your process has two steps:
    1. INTERVIEW: Ask 2-3 open-ended, exploratory questions to understand the user. Good topics include their hobbies, what they are most excited to use this application for, or what a perfect digital assistant would do for them.
    2. CONCLUDE: After you have asked your questions and received answers, you must end the conversation. To do this, your final response and ONLY your final response must be the special command: [END_ONBOARDING].
    """,
    "NAVIGATION_ASSISTANT": """
    You are an expert command processing AI. Your only job is to analyze the user's transcribed text and determine which of the following commands to issue. Respond with ONLY the single, most appropriate command name and nothing else.

    **Available Commands:**
    - 'GOTO_INTERVIEW_SCREEN'
    - 'GOTO_FEEDBACK_SCREEN'
    - 'EXPLAIN_INSTRUCTIONS'
    - 'UNKNOWN_COMMAND'

    **Examples of User Intent Mapping:**
    - User says: "I think I'm ready to give some mock interviews" -> Correct Command: 'GOTO_INTERVIEW_SCREEN'
    - User says: "let's start a practice session" -> Correct Command: 'GOTO_INTERVIEW_SCREEN'
    - User says: "I want to know about my past performance" -> Correct Command: 'GOTO_FEEDBACK_SCREEN'
    - User says: "can you show me my progress?" -> Correct Command: 'GOTO_FEEDBACK_SCREEN'
    - User says: "help" or "what can I do here?" -> Correct Command: 'EXPLAIN_INSTRUCTIONS'
    - User says: "what is the weather like today?" -> Correct Command: 'UNKNOWN_COMMAND'
    """,
    "SUMMARIZER": """
    You are a data analysis AI. The following is a conversation with a new user. Your sole task is to read the entire conversation and generate a JSON object summarizing the user's profile. The JSON should have three keys: "interests" (a list of strings), "goals" (a list of strings), and "challenges" (a list of strings). Output ONLY the raw JSON object and nothing else.
    """
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Interview Coach")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.app_state = None
        self.current_user = None
        self.current_frame = None
        self.conversation_history = []
        self.current_persona = None

        self.whisper_model, self.gemma_model = None, None
        self.piper_voice = None
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5
        self.microphone = sr.Microphone()

        self.show_welcome_screen()

    def show_frame(self, frame_class, **kwargs):
        if self.current_frame:
            self.current_frame.grid_forget()
        self.current_frame = frame_class(self, **kwargs)
        self.current_frame.grid(row=0, column=0, sticky="nsew")

    def show_welcome_screen(self):
        self.show_frame(WelcomeFrame, login_callback=self.login_user)

    def login_user(self, username):
        user_data = db.get_user_by_username(username)
        if user_data:
            self.current_user = user_data
            self.current_user['preferences'] = json.loads(self.current_user['preferences'])
            self.conversation_history = db.get_conversation_history(self.current_user['id'])
            self.transition_to_main_app()

    def logout_and_return_to_welcome(self):
        self.app_state = None
        self.current_user = None
        self.conversation_history = []
        self.title("AI Interview Coach")
        self.show_welcome_screen()
        self.current_frame.populate_profile_buttons()

    def transition_to_main_app(self):
        if self.current_user['role'] == 'admin':
            self.title("Admin Dashboard")
            self.show_frame(AdminDashboard, switch_profile_callback=self.logout_and_return_to_welcome)
        else:
            self.title(f"User: {self.current_user['username']}")
            self.show_frame(MainAppFrame)
            
            onboarding_complete = self.current_user['preferences'].get('onboarding_complete', False)
            if not onboarding_complete:
                self.app_state = "ONBOARDING"
                self.current_persona = "ONBOARDING_SPECIALIST"
                threading.Thread(target=self.initialize_models_and_start_onboarding, daemon=True).start()
            else:
                self.app_state = "NAVIGATION"
                self.current_persona = "NAVIGATION_ASSISTANT"
                welcome_message = f"Welcome back, {self.current_user['username']}!"
                last_screen = self.current_user['preferences'].get('last_screen', 'interview_screen')
                self.current_frame.show_screen(last_screen)
                threading.Thread(target=self.initialize_models_and_listen, args=(welcome_message,)).start()

    def speak(self, text):
        """Synthesizes and plays audio, ensuring it completes fully."""
        if not self.piper_voice or not text or not text.strip():
            return
        
        def audio_task():
            try:
                samplerate = self.piper_voice.config.sample_rate
                with sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
                    self.update_status("Speaking...")
                    for audio_chunk in self.piper_voice.synthesize(text):
                        stream.write(audio_chunk.audio_int16_array)
            except Exception as e:
                print(f"Piper TTS playback error: {e}")
        
        audio_thread = threading.Thread(target=audio_task)
        audio_thread.start()
        audio_thread.join()

    def play_audio_file(self, path):
        try:
            sound = AudioSegment.from_wav(path)
            play(sound)
        except FileNotFoundError:
            print(f"Warning: Audio file not found at {path}")
        except Exception as e:
            print(f"Error playing audio file {path}: {e}")

    def listen_after_prompt(self, prompt_text=""):
        if prompt_text:
            self.speak(prompt_text)
        
        self.play_audio_file(BEEP_SOUND_PATH)
        self.update_status("Listening...")
        
        try:
            with self.microphone as source:
                audio_data = self.recognizer.listen(source)
            self.update_status("Transcribing...")
            wav_data = audio_data.get_wav_data()
            temp_audio_path = Path("temp_audio.wav")
            with open(temp_audio_path, "wb") as f: f.write(wav_data)
            result = self.whisper_model.transcribe(str(temp_audio_path), fp16=False)
            user_input = result['text'].strip()
            os.remove(temp_audio_path)

            self.update_transcript(user_input)
            return user_input
        except sr.UnknownValueError:
            self.speak("I'm sorry, I didn't catch that. Let's try again.")
            return ""
        except Exception as e:
            print(f"An error occurred during listening: {e}")
            self.speak("Sorry, an error occurred while trying to listen.")
            return ""

    def _load_models(self):
        if not self.whisper_model:
            self.update_status("Loading speech model...")
            self.whisper_model = whisper.load_model("base.en")
        if not self.gemma_model:
            self.update_status("Loading Gemma AI...")
            self.gemma_model = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
        if not self.piper_voice:
            self.update_status("Loading voice model...")
            self.piper_voice = PiperVoice.load(PIPER_MODEL_PATH)

    def initialize_models_and_start_onboarding(self):
        self._load_models()
        self.update_status("Please listen to the instructions.")
        for path in ONBOARDING_AUDIO_FILES:
            self.play_audio_file(path)
        self.onboarding_listener()

    def initialize_models_and_listen(self, welcome_message=""):
        self._load_models()
        if welcome_message:
            self.speak(welcome_message)
        self.update_status("Ready for commands.")
        self.background_listener()

    def update_status(self, text):
        if isinstance(self.current_frame, MainAppFrame):
            self.current_frame.audio_status_label.configure(text=text)
    
    def update_transcript(self, text):
        """Safely updates the transcript label from any thread."""
        if isinstance(self.current_frame, MainAppFrame):
            self.after(0, lambda: self.current_frame.transcript_label.configure(text=f'You said: "{text}"'))

    def _process_gemma_response(self, full_prompt, max_tokens=150):
        """
        Takes a fully formatted prompt string and sends it to the LLM.
        """
        output = self.gemma_model(full_prompt, max_tokens=max_tokens, stop=["</s>", "[INST]", "User:", "Assistant:"], echo=False)
        return output['choices'][0]['text'].strip()

    def onboarding_listener(self):
        """Manages the conversational onboarding flow with correct prompt formatting."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        user_input = "" 
        while self.app_state == "ONBOARDING":
            if user_input:
                self.conversation_history.append({"role": "user", "content": user_input})
                db.add_message_to_history(self.current_user['id'], "user", user_input)

            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
            prompt_for_gemma = f"""
[INST]
{AI_PERSONAS[self.current_persona]}

Here is the conversation so far:
{history_str}

Based on the conversation, provide your next response as the assistant.
[/INST]
"""
            
            self.update_status("Gemma is thinking...")
            ai_response = self._process_gemma_response(prompt_for_gemma)

            if "[END_ONBOARDING]" in ai_response:
                self.execute_command("[END_ONBOARDING]")
                break

            self.conversation_history.append({"role": "assistant", "content": ai_response})
            db.add_message_to_history(self.current_user['id'], "assistant", ai_response)
            
            user_input = self.listen_after_prompt(prompt_text=ai_response)
            if not user_input:
                user_input = "..."
    
    def background_listener(self):
        """Listens for stateless navigation commands with the new, robust prompt."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while self.app_state == "NAVIGATION":
            user_text = self.listen_after_prompt()
            
            if user_text:
                self.update_status(f"Heard: '{user_text}'\n\nThinking...")

                prompt = f"""[INST]
                {AI_PERSONAS['NAVIGATION_ASSISTANT']}

                User Request: "{user_text}"

                Command:
                [/INST]"""

                command = self._process_gemma_response(prompt, max_tokens=30)
                print(f"DEBUG: Cleaned command from Gemma: '{command}'")
                self.execute_command(command)

    def execute_command(self, command: str):
        """Handles navigation, special, and help commands."""
        clean_command = command.strip().strip("'\"")

        if clean_command == "[END_ONBOARDING]":
            self.update_status("Finalizing your profile...")
            self.speak("Thank you. One moment while I set up your profile.")
            threading.Thread(target=self.summarize_and_conclude_onboarding, daemon=True).start()
            return
        
        self.update_status(f"Command: {clean_command}")
        
        if clean_command == "GOTO_INTERVIEW_SCREEN":
            self.speak("Okay, showing the Interview Screen.")
            if isinstance(self.current_frame, MainAppFrame):
                self.current_frame.show_screen("interview_screen")
        elif clean_command == "GOTO_FEEDBACK_SCREEN":
            self.speak("Okay, showing the Feedback Screen.")
            if isinstance(self.current_frame, MainAppFrame):
                self.current_frame.show_screen("feedback_screen")
        elif clean_command == "EXPLAIN_INSTRUCTIONS":
            self.speak("Of course. Here are the instructions again.")
            self.play_audio_file(ONBOARDING_AUDIO_FILES[1])
        else:
            self.speak("I'm sorry, I didn't understand that command.")

    def summarize_and_conclude_onboarding(self):
        """Fetches history, gets summary from LLM, and updates the database."""
        final_history = db.get_conversation_history(self.current_user['id'])
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_history])
        
        summarizer_prompt = f"""
        [INST]
        {AI_PERSONAS['SUMMARIZER']}

        CONVERSATION HISTORY:
        {history_text}
        [/INST]
        """
        
        self.update_status("Creating profile summary...")
        json_summary_str = self._process_gemma_response(summarizer_prompt)
        
        try:
            if json_summary_str.startswith("```json"):
                json_summary_str = json_summary_str[7:]
                if json_summary_str.endswith("```"):
                    json_summary_str = json_summary_str[:-3]
            
            profile_summary = json.loads(json_summary_str)
            updated_prefs = self.current_user['preferences']
            updated_prefs['onboarding_complete'] = True
            updated_prefs['profile_summary'] = profile_summary
            db.update_user_preferences(self.current_user['id'], updated_prefs)
            print("Successfully saved profile summary:", profile_summary)
        except json.JSONDecodeError as e:
            print(f"Error: LLM did not return valid JSON for summary. Error: {e}")
            print(f"Received: {json_summary_str}")
            updated_prefs = self.current_user['preferences']
            updated_prefs['onboarding_complete'] = True
            db.update_user_preferences(self.current_user['id'], updated_prefs)

        self.app_state = "NAVIGATION"
        self.current_persona = "NAVIGATION_ASSISTANT"
        self.update_status("Profile setup complete!")
        self.speak("Your profile is now set up. From now on, I'll be your navigation assistant. Just tell me which screen you'd like to go to.")
        
        self.background_listener()

if __name__ == "__main__":
    print("Application starting up...")
    db.initialize_database()
    app = App()
    app.mainloop()