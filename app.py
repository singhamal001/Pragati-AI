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

import prompts
import gemma_logic
import interview_analyzer
import feedback_manager

import interview_flow_manager

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

        self.whisper_model, self.gemma_model = None, None
        self.piper_voice = None
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.5
        self.microphone = sr.Microphone()

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
                if self.current_frame:
                    self.current_frame.show_screen(last_screen)

                def post_load_task():
                    self.listener_stop_flag = threading.Event()
                    threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()
                    
                    self.speak(welcome_message)
                    self.update_status("Ready for commands.")

                threading.Thread(target=lambda: (self._load_models(), post_load_task()), daemon=True).start()

    def speak(self, text):
        """Synthesizes and plays audio, ensuring it completes fully."""
        if not self.piper_voice or not text or not text.strip():
            return
        
        def audio_task():
            try:
                self.update_status("Speaking...")
                samplerate = self.piper_voice.config.sample_rate
                with sd.OutputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
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

        self.play_audio_file(BEEP_SOUND_PATH)
        self.update_status("Listening...")

        while True:
            try:
                self.recognizer.pause_threshold = pause_duration

                with self.microphone as source:
                    audio_data = self.recognizer.listen(
                        source,
                        timeout=initial_timeout,
                        phrase_time_limit=max_record_time
                    )

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
                self.speak("Sorry, an error occurred with the microphone.")
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
        """
        Thread-safe method to update the status label.
        It uses self.after() to schedule the UI update on the main thread.
        """
        if isinstance(self.current_frame, MainAppFrame):
            self.after(0, self.current_frame.audio_status_label.configure, {"text": text})
    
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
        # Clear any existing buttons
        for widget in self.current_frame.interview_list_frame.winfo_children():
            widget.destroy()

        # Get all past interviews for the current user
        interviews = feedback_manager.get_all_interviews_for_user(self.current_user['id'])

        if not interviews:
            ctk.CTkLabel(self.current_frame.interview_list_frame, text="No reports found.").pack(pady=10)
            return

        for interview in interviews:
            # Format the date nicely
            date_str = interview['timestamp'].split(" ")[0]
            button_text = f"{interview['interview_type']}\n{date_str}"
            
            button = ctk.CTkButton(
                self.current_frame.interview_list_frame,
                text=button_text,
                command=lambda i_id=interview['interview_id']: self.display_feedback_report(i_id)
            )
            button.pack(fill="x", padx=5, pady=5)

    # --- NEW: Function to display the details of a selected report ---
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
        
        # Update the textbox
        textbox = self.current_frame.report_display_textbox
        textbox.configure(state="normal") # Enable writing
        textbox.delete("1.0", "end")
        textbox.insert("1.0", formatted_text)
        textbox.configure(state="disabled") # Disable writing
        self.current_frame.discuss_button.configure(state="normal")
    
    def start_feedback_session(self):
        if self.interview_in_progress: # Use the same flag to prevent conflicts
            print("Cannot start feedback session while an interview is active.")
            return

        # 1. Get the report text from the UI
        report_text = self.current_frame.report_display_textbox.get("1.0", "end")
        if not report_text or len(report_text.strip()) < 20:
            print("Error: No report loaded to discuss.")
            self.speak("Please select a report from the list first.")
            return

        # 2. Set the application state
        self.app_state = "FEEDBACK_QA"
        print(f"DEBUG: App state changed to {self.app_state}")

        # 3. Stop the background listener
        if self.listener_stop_flag:
            self.listener_stop_flag.set()

        self.interview_in_progress = True # Reuse flag to lock UI

        # 4. Start the feedback thread
        threading.Thread(target=self._feedback_thread, args=(report_text,), daemon=True).start()

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
            user_question = self.listen_after_prompt(prompt_text=ai_response)
            
            if not user_question:
                self.speak("I'm sorry, I didn't catch that. Could you ask your question again?")
                continue

            # First, check for exit intent using our specialist
            exit_check_prompt = f"""[INST]{AI_PERSONAS['EXIT_DETECTOR']}
            User says: "{user_question}"[/INST]"""
            exit_decision = self._process_gemma_response(exit_check_prompt, max_tokens=10)

            if "YES_EXIT" in exit_decision:
                print("DEBUG: Exit intent detected.")
                self.speak("Of course. I'm glad I could help. Ending the feedback session now.")
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
            
            # We re-assign the new AI response to our main variable
            ai_response = self._process_gemma_response(qa_prompt, max_tokens=300)
            feedback_history.append({"role": "assistant", "content": ai_response})

        # --- Teardown Phase (Unchanged) ---
        self.after(0, lambda: self.current_frame.discuss_button.configure(state="normal"))
        self.after(0, lambda: self.current_frame.return_button.configure(state="normal"))
        self.app_state = "NAVIGATION"
        self.update_status("Ready for commands.")
        self.interview_in_progress = False
        self.listener_stop_flag = threading.Event()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()

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
    
    def background_listener(self, stop_event):
        """
        Listens for navigation commands in a dedicated, stoppable thread.

        This function contains its own listening logic and does NOT call the
        shared 'listen_after_prompt' function to avoid resource conflicts.
        """
        print("DEBUG: Background listener thread started.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while not stop_event.is_set():
            try:
                with self.microphone as source:
                    audio_data = self.recognizer.listen(source, timeout=1.5, phrase_time_limit=10)

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
                    self.execute_command(command)
                
                self.update_status("Ready for commands.")

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
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
                self.speak(ai_response)
                print("INFO: Interview concluded by AI's closing statement.")
                break

            user_answer = self.listen_after_prompt(prompt_text=ai_response)
            print(f"USER: {user_answer if user_answer else '<No input detected>'}")

            if not user_answer:
                self.speak("I'm sorry, I didn't get that. We can try that question again.")
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

        self.update_status("Interview finished. Analyzing...")
        self.speak("The interview is now complete. I'm analyzing your responses now...")
        
        analysis_results = interview_analyzer.run_full_analysis(
            self.gemma_model, self._process_gemma_response, interview_history, interview_type
        )
        if analysis_results:
            feedback_manager.save_feedback_to_db(self.current_user['id'], analysis_results)
            self.speak("Great session! Your detailed feedback report has been saved.")
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
            self.after(200, self.display_feedback_report, new_interview_id)
        # --------------------------------------------------------

        # This is now the ONLY call to restart the listener. It is correct.
        self.listener_stop_flag = threading.Event()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()


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

    # app.py -> inside the App class

    def summarize_and_conclude_onboarding(self):
        """Fetches history, gets summaries, and correctly restarts the listener."""
        final_history = db.get_conversation_history(self.current_user['id'])
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in final_history])
        
        json_summarizer_prompt = f"""
        [INST]
        {AI_PERSONAS['SUMMARIZER']}

        CONVERSATION HISTORY:
        {history_text}
        [/INST]
        """
        self.update_status("Creating profile summary...")
        json_summary_str = self._process_gemma_response(json_summarizer_prompt, max_tokens=500)
        
        narrative_summarizer_prompt = f"""
        [INST]
        {AI_PERSONAS['NARRATIVE_SUMMARIZER']}

        CONVERSATION HISTORY:
        {history_text}
        [/INST]
        """
        self.update_status("Creating narrative summary...")
        paragraph_summary_str = self._process_gemma_response(narrative_summarizer_prompt, max_tokens=250)
        
        try:
            if json_summary_str.startswith("```json"):
                json_summary_str = json_summary_str[7:]
                if json_summary_str.endswith("```"):
                    json_summary_str = json_summary_str[:-3]
            profile_summary_json = json.loads(json_summary_str)
            updated_prefs = self.current_user['preferences']
            updated_prefs['onboarding_complete'] = True
            updated_prefs['profile_summary'] = profile_summary_json
            updated_prefs['narrative_summary'] = paragraph_summary_str.strip()
            db.update_user_preferences(self.current_user['id'], updated_prefs)
            print("Successfully saved structured and narrative summaries.")
        except json.JSONDecodeError as e:
            print(f"Error: LLM did not return valid JSON for summary. Error: {e}")
            updated_prefs = self.current_user['preferences']
            updated_prefs['onboarding_complete'] = True
            db.update_user_preferences(self.current_user['id'], updated_prefs)

        self.app_state = "NAVIGATION"
        self.current_persona = "NAVIGATION_ASSISTANT"
        self.update_status("Profile setup complete!")
        self.speak("Your profile is now set up. From now on, I'll be your navigation assistant.")
        
        self.listener_stop_flag = threading.Event()
        threading.Thread(target=self.background_listener, args=(self.listener_stop_flag,), daemon=True).start()

if __name__ == "__main__":
    print("Application starting up...")
    db.initialize_database()
    app = App()
    app.mainloop()