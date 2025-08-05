# app.py

import customtkinter as ctk
import threading
import speech_recognition as sr
import whisper
from llama_cpp import Llama
import os
from pathlib import Path
import json
from datetime import datetime

from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

import prompts
import gemma_logic
import interview_analyzer
import feedback_manager

import interview_flow_manager

import database_manager as db
from ui_components import WelcomeFrame, AdminDashboard, MainAppFrame
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- Constants ---
MODEL_PATH = resource_path("./model/gemma-3n-e2b-it.Q2_K_M.gguf")
PIPER_MODEL_PATH = resource_path("./model/en_US-hfc_female-medium.onnx")

MAX_ONBOARDING_TURNS = 4

# --- Centralized Audio Path Manager ---
AUDIO_PATHS = {
    # Startup & Login
    "app_startup": resource_path("./assets/audio/other/app_startup.wav"),
    "login_success": resource_path("./assets/audio/other/login_success.wav"),
    "logout_confirmation": resource_path("./assets/audio/other/logout_confirmation.wav"),

    # Onboarding
    "onboarding_start": resource_path("./assets/audio/onboarding/onboarding_start.wav"),
    "onboarding_instructions_intro": resource_path("./assets/audio/onboarding/onboarding_instructions_intro.wav"),
    "onboarding_concluding": resource_path("./assets/audio/onboarding/onboarding_concluding.wav"),
    "onboarding_complete_transition": resource_path("./assets/audio/onboarding/onboarding_complete_transition.wav"),
    # Legacy instruction files for now
    "instructions_part1": resource_path("./assets/audio/other/instructions_part1.wav"),
    "instructions_part2": resource_path("./assets/audio/other/instructions_part2.wav"),


    # Navigation
    "nav_main_menu_prompt": resource_path("./assets/audio/navigation/nav_main_menu_prompt.wav"),
    "nav_unknown_command": resource_path("./assets/audio/navigation/nav_unknown_command.wav"),

    # Interview Flow
    "interview_screen_prompt": resource_path("./assets/audio/interview/interview_screen_prompt.wav"),
    "interview_starting": resource_path("./assets/audio/interview/interview_starting.wav"),
    "interview_ai_thinking": resource_path("./assets/audio/interview/interview_ai_thinking.wav"),
    "interview_no_input_detected": resource_path("./assets/audio/interview/interview_no_input_detected.wav"),
    "interview_ending": resource_path("./assets/audio/interview/interview_ending.wav"),
    "interview_analysis_starting": resource_path("./assets/audio/interview/interview_analysis_starting.wav"),
    "interview_analysis_complete": resource_path("./assets/audio/interview/interview_analysis_complete.wav"),

    # Feedback Flow
    "feedback_screen_prompt": resource_path("./assets/audio/feedback/feedback_screen_prompt.wav"),
    "feedback_list_reports_intro": resource_path("./assets/audio/feedback/feedback_list_reports_intro.wav"),
    "feedback_no_reports_found": resource_path("./assets/audio/feedback/feedback_no_reports_found.wav"),
    "feedback_report_selected": resource_path("./assets/audio/feedback/feedback_report_selected.wav"),
    "feedback_discussion_starting": resource_path("./assets/audio/feedback/feedback_discussion_starting.wav"),
    "feedback_session_ending": resource_path("./assets/audio/feedback/feedback_session_ending.wav"),
    "feedback_screen_prompt_guided": resource_path("./assets/audio/feedback/feedback_screen_prompt.wav"),
    "feedback_prompt_for_selection": resource_path("./assets/audio/feedback/feedback_prompt_for_selection.wav"),

    # Errors & System
    "error_microphone": resource_path("./assets/audio/error/error_microphone.wav"),
    "error_model_general": resource_path("./assets/audio/error/error_model_general.wav"),
    "error_file_not_found": resource_path("./assets/audio/error/error_file_not_found.wav"),
    "beep": resource_path("./assets/audio/other/beep.wav"),
}

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
    - 'START_BACKGROUND_INTERVIEW'
    - 'START_SALARY_INTERVIEW'    

    **Examples of User Intent Mapping:**
    - User says: "I think I'm ready to give some mock interviews" -> Correct Command: 'GOTO_INTERVIEW_SCREEN'
    - User says: "let's start a practice session" -> Correct Command: 'GOTO_INTERVIEW_SCREEN'
    - User says: "I want to know about my past performance" -> Correct Command: 'GOTO_FEEDBACK_SCREEN'
    - User says: "can you show me my progress?" -> Correct Command: 'GOTO_FEEDBACK_SCREEN'
    - User says: "help" or "what can I do here?" -> Correct Command: 'EXPLAIN_INSTRUCTIONS'
    - User says: "what is the weather like today?" -> Correct Command: 'UNKNOWN_COMMAND'
    - User says: "let's start a background interview" -> Correct Command: 'START_BACKGROUND_INTERVIEW'
    - User says: "begin the background check" -> Correct Command: 'START_BACKGROUND_INTERVIEW'
    - User says: "start the salary negotiation" -> Correct Command: 'START_SALARY_INTERVIEW'
    - User says: "I'm ready to talk about salary" -> Correct Command: 'START_SALARY_INTERVIEW'

    These examples are for your understanding and inference, this doesn't mean any rule based methodology, the major task is to understand what the user is trying to imply from his words.
    """,

    "SUMMARIZER": """
    You are a data analysis AI. The following is a conversation with a new user. Your sole task is to read the entire conversation and generate a JSON object summarizing the user's profile. The JSON should have three keys: "interests" (a list of strings), "goals" (a list of strings), and "challenges" (a list of strings). Output ONLY the raw JSON object and nothing else.
    """,
     "FEEDBACK_COACH": """
    You are Gemma, an encouraging and insightful AI career coach. Your task is to help a visually impaired user understand their interview feedback report and answer their questions about it.

    **CONTEXT:** The user has selected a past interview report. The full text of that report will be provided to you.

    **YOUR PROCESS:**
    1.  **INITIAL SUMMARY:** Your VERY FIRST response MUST be a high-level, conversational summary of the provided report. Start by highlighting 1-2 key strengths (what went well) and then 1-2 main areas for improvement (what to focus on next). Keep this initial summary concise.
    2.  **Q&A SESSION:** After your initial summary, the user will ask you questions. Answer them based ONLY on the information in the provided report text. Be supportive and provide actionable advice.
    3.  **CONCLUDE:** When the user indicates they are finished (e.g., "that's all," "thank you," "end session"), your final response, and ONLY your final response, MUST be the special command: `[END_FEEDBACK]`.

    Do not make up information not present in the report.
    """,
    "NARRATIVE_SUMMARIZER": """
    You are an expert summarization AI. Read the following conversation with a new user and generate a concise, one-paragraph summary (under 150 words) that captures the user's background, primary goals, and key challenges mentioned. This summary will be used as a quick human-readable reference. Respond with ONLY the paragraph summary and nothing else.
    """,
    "EXIT_DETECTOR": """
    You are a simple binary classification AI. Your only task is to determine if the user's statement expresses an intent to end the current conversation.

    RULES:
    - If the user's statement means they want to stop, leave, or are finished, respond with ONLY the keyword: `YES_EXIT`.
    - If the user is asking a question or making any other statement, respond with ONLY the keyword: `NO_EXIT`.

    EXAMPLES:
    - User says: "thank you, I'm done" -> Your Response: `YES_EXIT`
    - User says: "that's all for now" -> Your Response: `YES_EXIT`
    - User says: "what was my star score for question 3?" -> Your Response: `NO_EXIT`
    - User says: "can you explain that differently" -> Your Response: `NO_EXIT`
    """,
    "ORDINAL_SELECTOR": """
    You are a number parsing AI. Your only job is to find a number or position in the user's text and convert it to a zero-based index. The user is selecting from a list of {list_length} items.

    RULES:
    - "first", "one", "1" -> 0
    - "second", "two", "2" -> 1
    - "third", "three", "3" -> 2
    - "fourth", "four", "4" -> 3
    - "fifth", "five", "5" -> 4
    - "last", "latest", "most recent" -> {list_length_minus_one}

    If you find a valid number or position, respond with ONLY the numeric index.
    If you cannot determine a specific number, respond with the single word: "UNKNOWN".

    User says: "{user_text}"
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
        self.listener_stop_flag = None
        self.interview_in_progress = False

        self.feedback_listener_stop_event = None
        self.in_feedback_mode = False

        self.whisper_model, self.gemma_model = None, None
        self.piper_voice = None
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 2.0
        self.microphone = sr.Microphone()

        self.stop_listening_event = None

        threading.Thread(target=lambda: self.play_audio("app_startup"), daemon=True).start()

        self.show_welcome_screen()

    def _clear_chat_ui(self):
        for widget in self.current_frame.chat_history_frame.winfo_children():
            widget.destroy()

    def _add_message_to_chat_ui(self, role, text):
        if role == "assistant":
            label = ctk.CTkLabel(self.current_frame.chat_history_frame, text=text, wraplength=600, justify="left", anchor="w")
            label.pack(anchor="w", padx=10, pady=5)
        else:
            label = ctk.CTkLabel(self.current_frame.chat_history_frame, text=text, wraplength=600, justify="right", anchor="e", text_color="#24a0ed")
            label.pack(anchor="e", padx=10, pady=5)
        
        self.after(100, self.current_frame.chat_history_frame._parent_canvas.yview_moveto, 1.0)

    def show_frame(self, frame_class, **kwargs):
        if self.current_frame:
            self.current_frame.grid_forget()
        self.current_frame = frame_class(master=self, **kwargs)
        self.current_frame.grid(row=0, column=0, sticky="nsew")
    
    def _sanitize_for_speech(self, text: str) -> str:
        """
        Removes characters and formatting that are read awkwardly by TTS engines.
        """
        if not isinstance(text, str):
            return ""

        text = text.replace('*', '')
        text = text.replace('#', '')
        text = text.replace('`', '')

        text = text.replace(':', '.')

        return text.strip()

    def show_welcome_screen(self):
        self.show_frame(WelcomeFrame, login_callback=self.login_user)

    def login_user(self, username):
        user_data = db.get_user_by_username(username)
        if user_data:
            self.play_audio("login_success")
            self.current_user = user_data
            self.current_user['preferences'] = json.loads(self.current_user['preferences'])
            self.conversation_history = db.get_conversation_history(self.current_user['id'])
            self.transition_to_main_app()

    def logout_and_return_to_welcome(self):
        # --- NEW: Signal the background listener to stop ---
        if self.stop_listening_event:
            print("DEBUG: Setting stop event for listener thread.")
            self.stop_listening_event.set()
            self.stop_listening_event = None

        self.play_audio("logout_confirmation")
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
                last_screen = self.current_user['preferences'].get('last_screen', 'interview_screen')
                self.current_frame.show_screen(last_screen)
                
                # --- MODIFIED: Start the initialization thread without arguments ---
                threading.Thread(target=self.initialize_models_and_listen, daemon=True).start()

    def speak(self, text):
        """Synthesizes and plays audio, ensuring it completes fully."""
        if not self.piper_voice or not text or not text.strip():
            return
        
        def audio_task():
            try:
                self._show_speaking_indicator()
                self.update_status("Speaking...")
                samplerate = self.piper_voice.config.sample_rate
                with sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
                    for audio_chunk in self.piper_voice.synthesize(text):
                        stream.write(audio_chunk.audio_int16_array)
            except Exception as e:
                print(f"Piper TTS playback error: {e}")
            finally:
                self._hide_speaking_indicator()
        
        audio_thread = threading.Thread(target=audio_task)
        audio_thread.start()
        audio_thread.join()

    # --- NEW: Centralized audio player ---
    def play_audio(self, audio_key: str):
        """Plays an audio file by its logical name and logs the action."""
        # --- NEW: Debugging Logs ---
        print(f"AUDIO_PLAYER: Attempting to play '{audio_key}'...")
        try:
            path = AUDIO_PATHS[audio_key]
            sound = AudioSegment.from_wav(path)
            play(sound)
            print(f"AUDIO_PLAYER: Successfully played '{audio_key}'.")
        except KeyError:
            print(f"AUDIO_PLAYER_ERROR: Audio key '{audio_key}' not found in AUDIO_PATHS.")
        except FileNotFoundError:
            print(f"AUDIO_PLAYER_ERROR: File not found at path: {path}")
            # Fallback to prevent silence
            self.speak("A required audio file could not be found.")
        except Exception as e:
            print(f"AUDIO_PLAYER_ERROR: Could not play '{audio_key}'. Reason: {e}")

    def play_audio_file(self, path):
        """DEPRECATED but kept for compatibility. Plays a single audio file by its direct path."""
        try:
            sound = AudioSegment.from_wav(path)
            play(sound)
        except FileNotFoundError:
            print(f"Warning: Audio file not found at {path}")
            self.play_audio("error_file_not_found")
        except Exception as e:
            print(f"Error playing audio file {path}: {e}")


    def listen_after_prompt(self, prompt_text=""):
        """
        Plays a prompt, then enters a dedicated loop to wait for and record a user's full answer.
        This function now contains ALL the important timing settings to prevent interruptions.
        """
        if prompt_text:
            self.speak(prompt_text)

        # --- Let's define our timing parameters clearly here ---
        pause_duration = 1.5  
        max_record_time = 300
        initial_timeout = 30

        # --- For your visibility, let's print the settings we're using ---
        print(f"DEBUG: Listener settings: pause_threshold={pause_duration}s, phrase_limit={max_record_time}s")
        # --------------------------------------------------------------------

        self.play_audio("beep") # Use the new audio manager
        self.update_status("Listening...")

        try:
            self._show_speaking_indicator()
            while True:
                try:
                    self.recognizer.pause_threshold = pause_duration

                    with self.microphone as source:
                        audio_data = self.recognizer.listen(
                            source,
                            timeout=initial_timeout,
                            phrase_time_limit=max_record_time
                        )
                    
                    self._hide_speaking_indicator()
                    self.update_status("Transcribing...")
                    wav_data = audio_data.get_wav_data()
                    temp_audio_path = Path("temp_audio.wav")
                    with open(temp_audio_path, "wb") as f:
                        f.write(wav_data)

                    result = self.whisper_model.transcribe(str(temp_audio_path), fp16=False)
                    user_input = result['text'].strip()
                    os.remove(temp_audio_path)

                    if user_input:
                        self.update_transcript(user_input)
                        return user_input
                    else:
                        self.update_status("Listening...")
                        continue

                except sr.WaitTimeoutError:
                    self.update_status("Listening...")
                    continue
                except sr.UnknownValueError:
                    self.update_status("Listening...")
                    continue
                except Exception as e:
                    print(f"An unexpected error occurred during listening: {e}")
                    self.play_audio("error_microphone")
                    return ""
        finally:
            self._hide_speaking_indicator()


    def _load_models(self):
        if not self.whisper_model:
            self.update_status("Loading speech model...")
            whisper_model_path = resource_path("./model/base.en.pt")
            self.whisper_model = whisper.load_model(whisper_model_path)
        if not self.gemma_model:
            self.update_status("Loading Gemma AI...")
            self.gemma_model = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
        if not self.piper_voice:
            self.update_status("Loading voice model...")
            self.piper_voice = PiperVoice.load(PIPER_MODEL_PATH)

    def initialize_models_and_start_onboarding(self):
        self._load_models()
        
        self.play_audio("onboarding_start")
        self.play_audio("onboarding_instructions_intro")
        
        self.play_audio("instructions_part1")
        self.play_audio("instructions_part2")
        
        self.onboarding_listener()


    def initialize_models_and_listen(self):
        """Loads models and starts the normal command listener."""
        self._load_models()
        
        self.play_audio("nav_main_menu_prompt")
        if self.current_user:
            self.speak(f"Welcome back, {self.current_user['username']}")

        self.update_status("Ready for commands.")
        
        # --- NEW: Create and pass the stop_event to the thread ---
        self.stop_listening_event = threading.Event()
        threading.Thread(
            target=self.background_listener, 
            args=(self.stop_listening_event,), 
            daemon=True
        ).start()

    def enter_feedback_mode(self):
        """Stops the main listener and starts the dedicated feedback listener."""
        if self.in_feedback_mode: # Prevent this from running twice
            return
        print("DEBUG: Entering Feedback Mode.")
        self.in_feedback_mode = True

        # Stop the main navigation listener
        if self.stop_listening_event:
            print("DEBUG: Stopping main navigation listener.")
            self.stop_listening_event.set()
            self.stop_listening_event = None

        # Start the new feedback listener in a separate thread
        print("DEBUG: Starting feedback navigation listener.")
        self.feedback_listener_stop_event = threading.Event()
        threading.Thread(
            target=self.feedback_navigation_listener,
            args=(self.feedback_listener_stop_event,),
            daemon=True
        ).start()

    def _show_speaking_indicator(self):
        """Schedules the speaking indicator to appear and start animating."""
        if isinstance(self.current_frame, MainAppFrame):
            def show():
                self.current_frame.speaking_indicator.pack(padx=20, pady=(5, 0), fill="x")
                self.current_frame.speaking_indicator.start()
            self.after(0, show)

    def _hide_speaking_indicator(self):
        """Schedules the speaking indicator to stop and disappear."""
        if isinstance(self.current_frame, MainAppFrame):
            def hide():
                self.current_frame.speaking_indicator.stop()
                self.current_frame.speaking_indicator.pack_forget()
            self.after(0, hide)


    def exit_feedback_mode_if_active(self):
        """Stops the feedback listener and restarts the main one."""
        if not self.in_feedback_mode:
            return
        print("DEBUG: Exiting Feedback Mode.")
        self.in_feedback_mode = False

        # Stop the feedback listener
        if self.feedback_listener_stop_event:
            print("DEBUG: Stopping feedback navigation listener.")
            self.feedback_listener_stop_event.set()
            self.feedback_listener_stop_event = None

        # Restart the main navigation listener so the user can give commands again
        print("DEBUG: Restarting main navigation listener.")
        self.update_status("Ready for commands.")
        self.stop_listening_event = threading.Event()
        threading.Thread(
            target=self.background_listener,
            args=(self.stop_listening_event,),
            daemon=True
        ).start()

    def feedback_navigation_listener(self, stop_event):
        """
        A dedicated listener that fetches reports, announces them,
        and then listens for and processes the user's selection.
        """
        import time
        time.sleep(0.5)

        self.play_audio("feedback_screen_prompt_guided")
        self.current_report_list = feedback_manager.get_all_interviews_for_user(self.current_user['id'])

        if not self.current_report_list:
            self.play_audio("feedback_no_reports_found")
            self.after(0, self.exit_feedback_mode_if_active)
            return

        for i, report in enumerate(self.current_report_list):
            if stop_event.is_set(): return
            ordinal = self._number_to_ordinal(i + 1)
            date_obj = datetime.fromisoformat(report['timestamp'])
            date_str = date_obj.strftime("%B %dth")
            announcement = f"{ordinal}, a {report['interview_type']} interview from {date_str}."
            print(f"ANNOUNCING: {announcement}")
            self.speak(announcement)
            time.sleep(0.5)

        # --- SELECTION LOOP ---
        while not stop_event.is_set():
            self.play_audio("feedback_prompt_for_selection")
            user_choice_text = self.listen_after_prompt() # Re-uses your existing robust listener

            if not user_choice_text:
                self.speak("I'm sorry, I didn't catch that. Please say which report you'd like.")
                continue

            # Call the Ordinal Selector AI
            list_len = len(self.current_report_list)
            prompt = AI_PERSONAS["ORDINAL_SELECTOR"].format(
                list_length=list_len,
                list_length_minus_one=list_len - 1,
                user_text=user_choice_text
            )
            full_prompt = f"[INST]{prompt}[/INST]"
            index_str = self._process_gemma_response(full_prompt, max_tokens=5)

            try:
                selected_index = int(index_str)
                if 0 <= selected_index < list_len:
                    # Valid index found!
                    selected_report = self.current_report_list[selected_index]
                    interview_id_to_discuss = selected_report['interview_id']

                    # Confirm with the user
                    date_obj = datetime.fromisoformat(selected_report['timestamp'])
                    date_str = date_obj.strftime("%B %dth")
                    confirmation_prompt = f"Okay, discussing the {selected_report['interview_type']} interview from {date_str}. Is that correct?"
                    
                    user_confirmation = self.listen_after_prompt(prompt_text=confirmation_prompt)
                    
                    if user_confirmation and "yes" in user_confirmation.lower():
                        stop_event.set()
                        self.after(0, self.start_feedback_session, interview_id_to_discuss)
                        return
                    else:
                        self.speak("My mistake. Let's try again.")
                        
                else:
                    self.speak("I don't see a report with that number. Please try again.")
            except (ValueError, TypeError):
                self.speak("I didn't understand that selection. Please say, for example, 'the first one' or 'the last one'.")
                
        print("FEEDBACK LISTENER: Thread has successfully stopped.")

    def update_status(self, text):
        """
        Thread-safe method to update the status label.
        It uses self.after() to schedule the UI update on the main thread.
        """
        if isinstance(self.current_frame, MainAppFrame):
            self.after(0, self.current_frame.audio_status_label.configure, {"text": text})
    
    def _number_to_ordinal(self, n):
        """
        Converts a number (e.g., 1) to a 
        spoken ordinal (e.g., 'First').
        """
        if 1 <= n <= 10:
            ordinals = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
            return ordinals[n-1]
        else:
            return f"The {n}th"
        
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
    
    def populate_interview_list(self):
        for widget in self.current_frame.interview_list_frame.winfo_children():
            widget.destroy()

        interviews = feedback_manager.get_all_interviews_for_user(self.current_user['id'])

        # if not interviews:
        #     ctk.CTkLabel(self.current_frame.interview_list_frame, text="No reports found.").pack(pady=10)
        #     return
        # self.play_audio("feedback_list_reports_intro")
        for interview in interviews:
            date_str = interview['timestamp'].split(" ")[0]
            button_text = f"{interview['interview_type']}\n{date_str}"
            
            button = ctk.CTkButton(
                self.current_frame.interview_list_frame,
                text=button_text,
                command=lambda i_id=interview['interview_id']: self.display_feedback_report(i_id)
            )
            button.pack(fill="x", padx=5, pady=5)

    # --- Function to display the details of a selected report ---
    def display_feedback_report(self, interview_id: str):
        report_details = feedback_manager.get_report_details_by_interview_id(interview_id)

        if not report_details:
            formatted_text = "Error: Could not retrieve report details."
        else:
            # Format the details into a nice string
            header = f"Report for {report_details[0]['interview_type']} Interview\n"
            header += f"Date: {report_details[0]['timestamp']}\n"
            header += "="*50 + "\n\n"
            
            q_and_a = []
            for item in report_details:
                q_text = f"Q{item['question_number']}: {item['question_text']}\n"
                a_text = f"Your Answer: {item['answer_text']}\n\n"
                
                s_score = f"  - STAR Score: {item.get('star_score', 'N/A')}/10\n"
                s_reason = f"    Reason: {item.get('star_reason', 'N/A')}\n"
                k_score = f"  - Keywords Score: {item.get('keywords_score', 'N/A')}/10\n"
                k_reason = f"    Reason: {item.get('keywords_reason', 'N/A')}\n"
                p_score = f"  - Professionalism: {item.get('professionalism_score', 'N/A')}/10\n"
                p_reason = f"    Reason: {item.get('professionalism_reason', 'N/A')}\n"
                
                q_and_a.append(q_text + a_text + s_score + s_reason + k_score + k_reason + p_score + p_reason)

            formatted_text = header + "\n---\n\n".join(q_and_a)
        
        textbox = self.current_frame.report_display_textbox
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", formatted_text)
        textbox.configure(state="disabled")
        self.current_frame.discuss_button.configure(state="normal")
        self.play_audio("feedback_report_selected")
    
    def start_feedback_session(self, interview_id: str):
        """
        Starts the interactive feedback Q&A for a specific, voice-selected interview.
        """
        if self.interview_in_progress:
            print("Cannot start feedback session while another process is active.")
            return

        # 1. Get the full report details from the database
        report_details = feedback_manager.get_report_details_by_interview_id(interview_id)
        if not report_details:
            self.speak("I'm sorry, I couldn't retrieve the details for that report.")
            self.after(0, self.exit_feedback_mode_if_active)
            return

        # 2. Format the report into a single string for the AI
        header = f"Report for {report_details[0]['interview_type']} Interview from {report_details[0]['timestamp']}\n\n"
        q_and_a_text = "\n---\n".join(
            [f"Question {item['question_number']}: {item['question_text']}\nAnswer: {item['answer_text']}\nSTAR Score: {item.get('star_score', 'N/A')}, Reason: {item.get('star_reason', 'N/A')}" for item in report_details]
        )
        full_report_text = header + q_and_a_text

        # 3. Set the application state and lock the UI
        self.app_state = "FEEDBACK_QA"
        self.interview_in_progress = True # Reuse flag to prevent other actions
        print(f"DEBUG: App state changed to {self.app_state}")

        # 4. Play the introductory audio and start the feedback thread
        self.play_audio("feedback_discussion_starting")
        threading.Thread(target=self._feedback_thread, args=(full_report_text,), daemon=True).start()


    def _feedback_thread(self, report_text: str):
        """
        MODIFIED: This version has a simplified and corrected prompt structure
        to prevent AI confusion and restore correct conversational behavior.
        """
        # --- Setup Phase (Unchanged) ---
        self.after(0, lambda: self.current_frame.discuss_button.configure(state="disabled"))
        self.after(0, lambda: self.current_frame.return_button.configure(state="disabled"))
        self.update_status("Starting Feedback...")

        feedback_history = []
        
        # --- First Turn: Initial Summary (Unchanged, this part works well) ---
        initial_prompt = f"""
        [INST]
        {AI_PERSONAS['FEEDBACK_COACH']}
        Here is the full interview report to discuss:
        ---
        {report_text}
        ---
        Now, provide your initial summary of what went well and what can be improved.
        [/INST]
        """
        # We will use one consistent variable for the AI's response
        ai_response = self._process_gemma_response(initial_prompt, max_tokens=300)
        feedback_history.append({"role": "assistant", "content": ai_response})
        
        # --- Main Q&A Loop (CORRECTED) ---
        while True:
            # The last AI response is spoken to prompt the user
            user_question = self.listen_after_prompt(prompt_text=self._sanitize_for_speech(ai_response))
            
            if not user_question:
                self.speak("I'm sorry, I didn't catch that. Could you ask your question again?")
                continue

            # First, check for exit intent using our specialist
            exit_check_prompt = f"""[INST]{AI_PERSONAS['EXIT_DETECTOR']}
            User says: "{user_question}"[/INST]"""
            exit_decision = self._process_gemma_response(exit_check_prompt, max_tokens=10)

            if "YES_EXIT" in exit_decision:
                print("DEBUG: Exit intent detected.")
                self.play_audio("feedback_session_ending")
                break

            # If not exiting, add the user's question to the history
            feedback_history.append({"role": "user", "content": user_question})
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in feedback_history])
            
            # --- THIS IS THE SIMPLIFIED AND CORRECTED PROMPT ---
            # It clearly restates the persona and the goal in every turn.
            qa_prompt = f"""
            [INST]
            You are Gemma, the helpful Feedback Coach.
            The user is asking questions about the following report:
            ---
            {report_text}
            ---
            Here is the conversation so far:
            {history_str}
            ---
            Based on the user's last question, provide a helpful and encouraging answer.
            [/INST]
            """
            self.play_audio("interview_ai_thinking")
            ai_response = self._process_gemma_response(qa_prompt, max_tokens=300)
            feedback_history.append({"role": "assistant", "content": ai_response})

        # --- Teardown Phase (Unchanged) ---
        self.after(0, lambda: self.current_frame.discuss_button.configure(state="normal"))
        self.after(0, lambda: self.current_frame.return_button.configure(state="normal"))
        self.app_state = "NAVIGATION"
        self.update_status("Ready for commands.")
        self.interview_in_progress = False
        self.listener_stop_flag = threading.Event()
        self.exit_feedback_mode_if_active()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()

    def onboarding_listener(self):
        """Manages the conversational onboarding flow with a fixed number of turns."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # --- NEW: Initialize turn counter ---
        turn_counter = 0
        
        user_input = "" 
        
        while self.app_state == "ONBOARDING":
            if user_input and user_input != "...":
                self.conversation_history.append({"role": "user", "content": user_input})
                db.add_message_to_history(self.current_user['id'], "user", user_input)
                
                # --- NEW: Increment counter after a valid user response ---
                turn_counter += 1

            # --- NEW: Check if the turn limit has been reached ---
            if turn_counter >= MAX_ONBOARDING_TURNS:
                print(f"DEBUG: Reached max turns ({turn_counter}). Forcing end of onboarding.")
                # We use the [END_ONBOARDING] command to trigger the existing summarization flow
                self.execute_command("[END_ONBOARDING]")
                break # Exit the loop

            # Build the prompt with proper multi-turn formatting
            prompt_for_gemma = f"[INST] {AI_PERSONAS[self.current_persona]} [/INST]"
            if not self.conversation_history:
                 prompt_for_gemma += " Assistant:"
            else:
                for msg in self.conversation_history:
                    if msg['role'] == 'assistant':
                        prompt_for_gemma += f" {msg['content']}"
                    elif msg['role'] == 'user':
                        prompt_for_gemma += f"\n[INST] {msg['content']} [/INST]"

            self.update_status("Gemma is thinking...")
            ai_response = self._process_gemma_response(prompt_for_gemma)

            # The AI-driven [END_ONBOARDING] is now a fallback, not the primary method
            if "[END_ONBOARDING]" in ai_response:
                self.execute_command("[END_ONBOARDING]")
                break

            self.conversation_history.append({"role": "assistant", "content": ai_response})
            db.add_message_to_history(self.current_user['id'], "assistant", ai_response)
            
            user_input = self.listen_after_prompt(prompt_text=self._sanitize_for_speech(ai_response))
            
            if not user_input:
                self.conversation_history.pop()
                db.remove_last_message(self.current_user['id'])
                user_input = "..."
    
    def background_listener(self, stop_event):
        """
        Listens for navigation commands in a dedicated, stoppable thread.
        This version includes a beep to prompt the user to speak.
        """

        print("DEBUG: Background listener thread started.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while not stop_event.is_set():
            try:
                self.update_status("Ready for command...")
                self.play_audio("beep")

                with self.microphone as source:
                    audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                if stop_event.is_set():
                    break

                self.update_status("Transcribing command...")
                wav_data = audio_data.get_wav_data()
                temp_audio_path = Path("temp_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(wav_data)

                result = self.whisper_model.transcribe(str(temp_audio_path), fp16=False)
                user_text = result['text'].strip()
                os.remove(temp_audio_path)

                if user_text:
                    self.update_transcript(user_text)
                    self.update_status(f"Heard: '{user_text}'\n\nThinking...")

                    prompt = f"""[INST]
                    {AI_PERSONAS['NAVIGATION_ASSISTANT']}

                    User Request: "{user_text}"

                    Command:
                    [/INST]"""

                    command = self._process_gemma_response(prompt, max_tokens=30)
                    print(f"DEBUG: Cleaned command from Gemma: '{command}'")
                    self.after(0, self.execute_command, command)
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                self.update_status("Could not understand audio.")
                continue
            except Exception as e:
                print(f"An error occurred in background_listener: {e}")
                import time
                time.sleep(1)

        print("DEBUG: Background listener thread has successfully stopped.")

    def start_interview_session(self, interview_type):
        if self.interview_in_progress:
            return

        self._clear_chat_ui()

        if self.listener_stop_flag:
            self.listener_stop_flag.set()

        self.interview_in_progress = True
        threading.Thread(target=self._interview_thread, args=(interview_type,), daemon=True).start()


    def _interview_thread(self, interview_type):
        """Manages the entire interview flow, from start to analysis."""
        
        # --- Setup Phase ---
        # 1. Set a LONG pause threshold suitable for interview answers
        self.recognizer.pause_threshold = 2.5
        print(f"DEBUG: Mic pause_threshold set to {self.recognizer.pause_threshold} for interview.")

        self.after(0, lambda: self.current_frame.background_button.configure(state="disabled"))
        self.after(0, lambda: self.current_frame.salary_button.configure(state="disabled"))
        
        self.update_status(f"Starting {interview_type} Interview...")
        self.speak(f"Okay, let's begin the {interview_type} interview.")

        interview_history = []
        prompt_template = prompts.BACKGROUND_INTERVIEW_PROMPT if interview_type == "Background" else prompts.SALARY_NEGOTIATION_PROMPT
        
        turn_count = 0
        
        # --- Using a 'while True' loop managed by our flow controller ---
        while True:
            turn_count += 1
            print(f"\n--- Turn {turn_count} ---")

            self.play_audio("interview_ai_thinking")

            ai_response = gemma_logic.get_interview_response(self.gemma_model, self._process_gemma_response, interview_history, prompt_template)
            
            # Sanitize response from potential markdown
            if ai_response.startswith("```"):
                ai_response = ai_response.strip("` \n")

            if not ai_response: # Handle case where model returns empty string
                print("WARNING: Model returned empty response. Attempting to conclude.")
                self.speak("It seems we've reached a good stopping point. Thank you for your time.")
                break

            print(f"AI: {ai_response}")
            self.after(0, self._add_message_to_chat_ui, "assistant", ai_response)
            interview_history.append({"role": "assistant", "content": ai_response})
            
            # Check for natural conclusion phrases from the AI
            conclusion_phrases = ["thank you for your time", "we'll be in touch", "end the simulation", "conclude our discussion"]
            if any(phrase in ai_response.lower() for phrase in conclusion_phrases):
                self.speak(self._sanitize_for_speech(ai_response))
                print("INFO: Interview concluded by AI's closing statement.")
                break

            user_answer = self.listen_after_prompt(prompt_text=self._sanitize_for_speech(ai_response))
            print(f"USER: {user_answer if user_answer else '<No input detected>'}")

            if not user_answer:
                self.play_audio("interview_no_input_detected")
                interview_history.pop() # Remove the last AI question
                turn_count -= 1 # Don't count this as a turn
                continue
            
            self.after(0, self._add_message_to_chat_ui, "user", user_answer)
            interview_history.append({"role": "user", "content": user_answer})
            
            # --- USE THE FLOW MANAGER TO CHECK IF WE SHOULD END ---
            should_end, reason = interview_flow_manager.should_end_interview(interview_history, interview_type, turn_count)
            if should_end:
                print(f"INFO: Ending interview. Reason: {reason}")
                self.speak("Okay, that seems like a good place to stop. Thank you.")
                break

            if turn_count >= 12: # Final safety break
                print("INFO: Ending interview due to reaching max turn limit.")
                self.speak("We've covered a lot today, so let's wrap up there. Thank you.")
                break
        self.play_audio("interview_ending")
        self.update_status("Interview finished. Analyzing...")
        self.play_audio("interview_analysis_starting")
        
        analysis_results = interview_analyzer.run_full_analysis(
            self.gemma_model, self._process_gemma_response, interview_history, interview_type
        )
        if analysis_results:
            feedback_manager.save_feedback_to_db(self.current_user['id'], analysis_results)
            self.play_audio("interview_analysis_complete")
        else:
            self.speak("There was an issue generating the analysis, so no report was saved.")

        # --- Teardown Phase ---
        # 2. Restore the SHORT pause threshold for command listening
        self.recognizer.pause_threshold = 1.0 
        print(f"DEBUG: Mic pause_threshold restored to {self.recognizer.pause_threshold} for commands.")

        self.after(0, lambda: self.current_frame.background_button.configure(state="normal"))
        self.after(0, lambda: self.current_frame.salary_button.configure(state="normal"))
        
        self.app_state = "NAVIGATION"
        self.update_status("Ready for commands.")
        print(f"DEBUG: App state changed back to {self.app_state}. Restarting background listener.")
        
        self.interview_in_progress = False

        # --- Automatically switch to the feedback screen ---
        if analysis_results:
            new_interview_id = analysis_results[0].interview_id
            self.after(0, self.current_frame.show_screen, "feedback_screen")
            self.after(200, self.display_feedback_report, str(new_interview_id))
        # --------------------------------------------------------

        # This is now the ONLY call to restart the listener. It is correct.
        self.listener_stop_flag = threading.Event()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()


    def execute_command(self, command: str):
        clean_command = command.strip().strip("'\"")

        if clean_command == "[END_ONBOARDING]":
            threading.Thread(target=self.summarize_and_conclude_onboarding, daemon=True).start()
            return
        
        self.update_status(f"Command: {clean_command}")
        
        if clean_command == "GOTO_INTERVIEW_SCREEN":
            if isinstance(self.current_frame, MainAppFrame):
                self.current_frame.show_screen("interview_screen")
        elif clean_command == "GOTO_FEEDBACK_SCREEN":
            if isinstance(self.current_frame, MainAppFrame):
                self.current_frame.show_screen("feedback_screen")
        elif clean_command == "EXPLAIN_INSTRUCTIONS":
            self.speak("Of course. Here are the instructions again.")
            self.play_audio("instructions_part2")
        elif clean_command == "START_BACKGROUND_INTERVIEW":
            if isinstance(self.current_frame, MainAppFrame):
                self.start_interview_session("Background")    
        elif clean_command == "START_SALARY_INTERVIEW":
            if isinstance(self.current_frame, MainAppFrame):
                self.start_interview_session("Salary Negotiation")
        else:
            self.play_audio("nav_unknown_command")

    def summarize_and_conclude_onboarding(self):
        """Fetches history, gets summary from LLM, and updates the database."""
        self.play_audio("onboarding_concluding")
        
        final_history = db.get_conversation_history(self.current_user['id'])
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_history])
        summarizer_prompt = f"[INST]\n{AI_PERSONAS['SUMMARIZER']}\n\nCONVERSATION HISTORY:\n{history_text}\n[/INST]"
        
        self.update_status("Creating profile summary...")
        json_summary_str = self._process_gemma_response(summarizer_prompt)
        
        try:
            if json_summary_str.startswith("```json"):
                json_summary_str = json_summary_str[7:-3]
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

        # Final hand-off
        self.app_state = "NAVIGATION"
        self.current_persona = "NAVIGATION_ASSISTANT"
        self.update_status("Profile setup complete!")
        
        self.play_audio("onboarding_complete_transition")
        
        self.listener_stop_flag = threading.Event()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()

if __name__ == "__main__":
    print("Application starting up...")
    db.initialize_database()
    app = App()
    app.mainloop()