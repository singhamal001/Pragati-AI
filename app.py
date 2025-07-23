import customtkinter as ctk
import threading
import pyaudio
import wave
from playsound import playsound

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Modern Screen & Audio Navigator")
        self.geometry("800x600")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Audio Recording Attributes ---
        self.is_recording = False
        self.audio_thread = None
        self.audio_filename = "temp_recording.wav"

        # --- Layout Configuration ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsw")
        self.sidebar_label = ctk.CTkLabel(self.sidebar_frame, text="Controls", font=("Roboto", 24, "bold"))
        self.sidebar_label.pack(padx=20, pady=(20, 10))

        # --- Audio Control Section in Sidebar ---
        self.audio_label = ctk.CTkLabel(self.sidebar_frame, text="Audio Recorder", font=("Roboto", 16))
        self.audio_label.pack(padx=20, pady=(20, 5))
        
        self.record_button = ctk.CTkButton(self.sidebar_frame, text="Record", command=self.start_recording)
        self.record_button.pack(padx=20, pady=10)

        self.stop_button = ctk.CTkButton(self.sidebar_frame, text="Stop & Play", command=self.stop_recording, state="disabled")
        self.stop_button.pack(padx=20, pady=10)

        self.audio_status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Idle")
        self.audio_status_label.pack(padx=20, pady=10)


        # --- Screen Container (holds all screens) ---
        self.main_container = ctk.CTkFrame(self, corner_radius=10)
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        # ... [The rest of your screen setup code remains unchanged]
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.screen1_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen1_frame, text="Screen 1", font=("Roboto", 32, "bold")).pack(pady=50)
        self.screen2_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen2_frame, text="Screen 2", font=("Roboto", 32, "bold")).pack(pady=50)
        self.screen3_frame = ctk.CTkFrame(self.main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.screen3_frame, text="Screen 3", font=("Roboto", 32, "bold")).pack(pady=50)
        self.show_screen(self.screen1_frame)
        
        # --- Control Bar at the bottom ---
        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.grid(row=1, column=1, padx=20, pady=(0, 20), sticky="ew")
        self.input_entry = ctk.CTkEntry(self.control_frame, placeholder_text="Enter 'Screen1', 'Screen2', etc.")
        self.input_entry.pack(side="left", padx=15, pady=10, fill="x", expand=True)
        self.go_button = ctk.CTkButton(self.control_frame, text="Go", width=50, command=self.switch_screen)
        self.go_button.pack(side="left", padx=10, pady=10)
        self.status_label = ctk.CTkLabel(self.control_frame, text="")
        self.status_label.pack(side="left", padx=10, pady=10)

    # --- Audio Functions ---

    def start_recording(self):
        """Starts the audio recording process in a separate thread."""
        self.is_recording = True
        self.audio_status_label.configure(text="Status: Recording...")
        self.record_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # We use a thread to prevent the GUI from freezing during recording
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def stop_recording(self):
        """Stops the audio recording and triggers playback."""
        if self.is_recording:
            self.is_recording = False
            self.audio_status_label.configure(text="Status: Processing...")
            self.stop_button.configure(state="disabled")

    def record_audio(self):
        """The actual audio recording logic that runs in a thread."""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        
        while self.is_recording:
            data = stream.read(1024)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the recorded audio to a .wav file
        with wave.open(self.audio_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))
            
        # After saving, start playback in another thread
        playback_thread = threading.Thread(target=self.play_audio)
        playback_thread.start()

    def play_audio(self):
        """Plays the recorded audio file and resets the UI."""
        self.audio_status_label.configure(text="Status: Playing...")
        try:
            playsound(self.audio_filename)
        except Exception as e:
            self.audio_status_label.configure(text=f"Error: {e}")

        # Reset the UI buttons and status after playback is complete
        self.audio_status_label.configure(text="Status: Idle")
        self.record_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    # --- Screen Switching Functions (Unchanged) ---
    def show_screen(self, screen_frame):
        self.screen1_frame.grid_forget()
        self.screen2_frame.grid_forget()
        self.screen3_frame.grid_forget()
        screen_frame.grid(row=0, column=0, sticky="nsew")

    def switch_screen(self):
        user_input = self.input_entry.get().strip().lower()
        if user_input == "screen1": self.show_screen(self.screen1_frame)
        elif user_input == "screen2": self.show_screen(self.screen2_frame)
        elif user_input == "screen3": self.show_screen(self.screen3_frame)
        self.input_entry.delete(0, 'end')

if __name__ == "__main__":
    app = App()
    app.mainloop()