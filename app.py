import customtkinter as ctk
import threading
import speech_recognition as sr
import whisper
from llama_cpp import Llama
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play

# --- Constants ---
# Path to the GGUF model you downloaded
MODEL_PATH = "./models/gemma-3n-e2b-it.Q2_K_M.gguf"
ERROR_SOUND_PATH = "./assets/error.wav"
SYSTEM_PROMPT = """
You are an expert command processing AI for a desktop application. Your only job is to analyze the user's text and determine which of the following commands to issue.

The available commands are:
- 'GOTO_SCREEN_1'
- 'GOTO_SCREEN_2'
- 'GOTO_SCREEN_3'
- 'UNKNOWN_COMMAND'

Analyze the user's request and respond with ONLY the single, most appropriate command name from the list above. Do not add any explanation or conversational text.
"""

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Gemma Voice Navigator")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Layout Configuration ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsw")
        self.sidebar_label = ctk.CTkLabel(self.sidebar_frame, text="Controls", font=("Roboto", 24, "bold"))
        self.sidebar_label.pack(padx=20, pady=(20, 10))
        
        self.audio_status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Initializing...", wraplength=130, font=("Roboto", 16))
        self.audio_status_label.pack(padx=20, pady=20)

        # --- Screen Container ---
        self.main_container = ctk.CTkFrame(self, corner_radius=10)
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        self.screen1_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen1_frame, text="Screen 1", font=("Roboto", 32, "bold")).pack(pady=50)
        self.screen2_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen2_frame, text="Screen 2", font=("Roboto", 32, "bold")).pack(pady=50)
        self.screen3_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen3_frame, text="Screen 3", font=("Roboto", 32, "bold")).pack(pady=50)
        self.show_screen(self.screen1_frame)
        
        # --- Model Initialization ---
        self.whisper_model = None
        self.gemma_model = None
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.microphone = sr.Microphone()
        
        init_thread = threading.Thread(target=self.initialize_models, daemon=True)
        init_thread.start()

    def initialize_models(self):
        """Loads the heavy AI models in the background."""
        self.update_status("Loading Whisper...")
        self.whisper_model = whisper.load_model("base.en")
        
        self.update_status("Loading Gemma...")
        self.gemma_model = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
        
        self.update_status("Ready to Listen!")
        
        listener_thread = threading.Thread(target=self.background_listener, daemon=True)
        listener_thread.start()

    def update_status(self, text):
        """Safely updates the status label from any thread."""
        self.after(0, lambda: self.audio_status_label.configure(text=text))

    def background_listener(self):
        """The main loop that continuously listens for commands."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            self.update_status("Listening...")
            try:
                with self.microphone as source:
                    audio_data = self.recognizer.listen(source)
                
                self.update_status("Transcribing...")
                wav_data = audio_data.get_wav_data()
                temp_audio_path = Path("temp_audio.wav")
                with open(temp_audio_path, "wb") as f:
                    f.write(wav_data)
                
                result = self.whisper_model.transcribe(str(temp_audio_path))
                user_text = result['text'].strip()
                os.remove(temp_audio_path)

                if not user_text:
                    continue

                self.update_status(f"Heard: '{user_text}'\n\nThinking...")
                prompt = f"[INST] {SYSTEM_PROMPT} User text: '{user_text}' [/INST]"
                
                output = self.gemma_model(prompt, max_tokens=15, stop=["</s>", "[INST]"], echo=False)
                command = output['choices'][0]['text'].strip().strip("'\"")

                print(f"DEBUG: Cleaned command from Gemma: '{command}'") 

                self.execute_command(command)

            except sr.UnknownValueError:
                continue
            except Exception as e:
                self.update_status(f"Error: {e}")
                self.after(2000)

    def execute_command(self, command: str):
        """Analyzes the command from Gemma and switches screens."""
        self.update_status(f"Command: {command}")
        
        if command == "GOTO_SCREEN_1":
            self.show_screen(self.screen1_frame)
            self.update_status("Switched to Screen 1")
        elif command == "GOTO_SCREEN_2":
            self.show_screen(self.screen2_frame)
            self.update_status("Switched to Screen 2")
        elif command == "GOTO_SCREEN_3":
            self.show_screen(self.screen3_frame)
            self.update_status("Switched to Screen 3")
        else: # Handles UNKNOWN_COMMAND
            self.update_status("Command not recognized. Please try again.")
            # --- Use pydub for stable, blocking audio playback ---
            try:
                error_sound = AudioSegment.from_wav(ERROR_SOUND_PATH)
                play(error_sound) # This call blocks until the sound is done
            except FileNotFoundError:
                print(f"Warning: Audio file not found at {ERROR_SOUND_PATH}")
            except Exception as e:
                print(f"Error playing sound with pydub: {e}")

    def show_screen(self, screen_frame):
        """Hides all screens and then shows the selected one."""
        self.screen1_frame.grid_forget()
        self.screen2_frame.grid_forget()
        self.screen3_frame.grid_forget()
        screen_frame.grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.mainloop()