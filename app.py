import customtkinter as ctk
import threading
import speech_recognition as sr

# --- Constants ---
WAKE_WORD = "pragati"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Voice Activated Navigator")
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

        # --- Audio Status Section in Sidebar ---
        self.audio_label = ctk.CTkLabel(self.sidebar_frame, text="Voice Control", font=("Roboto", 16))
        self.audio_label.pack(padx=20, pady=(20, 5))
        
        self.audio_status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Initializing...", wraplength=130)
        self.audio_status_label.pack(padx=20, pady=10)

        # --- Screen Container (holds all screens) ---
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
        
        # --- Start the background listener ---
        listener_thread = threading.Thread(target=self.listen_for_wake_word, daemon=True)
        listener_thread.start()

    def update_status(self, text):
        """Safely updates the status label from a background thread."""
        self.audio_status_label.configure(text=text)

    def listen_for_wake_word(self):
        """Runs in a background thread to listen for the wake word and commands."""
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 1.0 # Pause of 1 second is enough to mark end of command
        microphone = sr.Microphone()

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            self.after(0, lambda: self.update_status(f"Listening for '{WAKE_WORD}'..."))

        while True:
            with microphone as source:
                try:
                    audio = recognizer.listen(source)
                    text = recognizer.recognize_google(audio).lower()
                    
                    if WAKE_WORD in text:
                        self.after(0, lambda: self.update_status("Wake word detected! Listening for command..."))
                        
                        # Listen for the actual command
                        command_audio = recognizer.listen(source)
                        command_text = recognizer.recognize_google(command_audio).lower()
                        
                        # Process the command
                        self.after(0, lambda: self.process_command(command_text))

                except sr.UnknownValueError:
                    # This is expected if silence or noise is detected, just continue
                    self.after(0, lambda: self.update_status(f"Listening for '{WAKE_WORD}'..."))
                    continue
                except sr.RequestError as e:
                    self.after(0, lambda: self.update_status(f"API Error: {e}"))
                    break

    def process_command(self, text):
        """Analyzes the command text and switches screens."""
        self.update_status(f"Heard: '{text}'")
        
        # Check for keywords to switch screens
        if "screen 1" in text or "scream 1" in text: # Added "scream" for common misinterpretations
            self.show_screen(self.screen1_frame)
            self.update_status("Switched to Screen 1")
        elif "screen 2" in text or "scream 2" in text:
            self.show_screen(self.screen2_frame)
            self.update_status("Switched to Screen 2")
        elif "screen 3" in text or "scream 3" in text:
            self.show_screen(self.screen3_frame)
            self.update_status("Switched to Screen 3")
        else:
            self.update_status("Command not recognized.")

        # After a short delay, go back to listening for the wake word
        self.after(2000, lambda: self.update_status(f"Listening for '{WAKE_WORD}'..."))

    def show_screen(self, screen_frame):
        """Hides all screens and then shows the selected one."""
        self.screen1_frame.grid_forget()
        self.screen2_frame.grid_forget()
        self.screen3_frame.grid_forget()
        screen_frame.grid(row=0, column=0, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.mainloop()