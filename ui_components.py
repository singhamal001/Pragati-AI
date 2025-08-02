# ui_components.py

import customtkinter as ctk
import database_manager as db

class WelcomeFrame(ctk.CTkFrame):
    def __init__(self, master, login_callback):
        super().__init__(master, corner_radius=0)
        self.login_callback = login_callback

        self.grid_columnconfigure(0, weight=1)
        self.welcome_label = ctk.CTkLabel(self, text="Select Your Profile", font=("Roboto", 28, "bold"))
        self.welcome_label.grid(row=0, column=0, pady=(100, 30), padx=20)
        
        self.buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.buttons_frame.grid(row=1, column=0, sticky="ew")
        self.buttons_frame.grid_columnconfigure(0, weight=1)
        
        self.populate_profile_buttons()

    def populate_profile_buttons(self):
        """Clears and rebuilds the login buttons."""
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

        profiles = db.get_all_users()
        for i, profile in enumerate(profiles):
            button_text = f"{profile['username']} (ID: {profile['id']})"
            button = ctk.CTkButton(
                self.buttons_frame,
                text=button_text,
                font=("Roboto", 18),
                command=lambda name=profile['username']: self.login_callback(name)
            )
            button.grid(row=i, column=0, pady=10, padx=150, sticky="ew")

class AdminDashboard(ctk.CTkFrame):
    def __init__(self, master, switch_profile_callback):
        super().__init__(master)
        self.switch_profile_callback = switch_profile_callback

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        ctk.CTkLabel(self.sidebar, text="Admin Menu", font=("Roboto", 20, "bold")).pack(pady=20)
        ctk.CTkButton(self.sidebar, text="Switch Profile", command=self.switch_profile_callback).pack(pady=10, padx=20, fill="x")

        content_area = ctk.CTkFrame(self, fg_color="transparent")
        content_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        content_area.grid_columnconfigure((0, 1), weight=1)
        content_area.grid_rowconfigure(0, weight=1)

        add_user_frame = ctk.CTkFrame(content_area)
        add_user_frame.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="nsew")
        add_user_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(add_user_frame, text="Add New User", font=("Roboto", 20, "bold")).grid(row=0, column=0, padx=10, pady=10)
        ctk.CTkLabel(add_user_frame, text="Name:").grid(row=1, column=0, padx=10, pady=(10,0), sticky="w")
        self.name_entry = ctk.CTkEntry(add_user_frame, placeholder_text="Enter name...")
        self.name_entry.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkLabel(add_user_frame, text="Age:").grid(row=3, column=0, padx=10, pady=(10,0), sticky="w")
        self.age_entry = ctk.CTkEntry(add_user_frame, placeholder_text="Enter age...")
        self.age_entry.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(add_user_frame, text="Add User", command=self.add_user_action).grid(row=5, column=0, padx=10, pady=20)

        manage_users_frame = ctk.CTkFrame(content_area)
        manage_users_frame.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        manage_users_frame.grid_columnconfigure(0, weight=1)
        manage_users_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(manage_users_frame, text="Manage Users", font=("Roboto", 20, "bold")).grid(row=0, column=0, padx=10, pady=10)
        self.user_list_frame = ctk.CTkScrollableFrame(manage_users_frame, label_text="Existing Users")
        self.user_list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.user_list_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(content_area, text="", font=("Roboto", 16))
        self.status_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.refresh_user_list()

    def add_user_action(self):
        name = self.name_entry.get().strip()
        age_str = self.age_entry.get().strip()
        if not name or not age_str:
            self.status_label.configure(text="Error: Name and age cannot be empty.", text_color="red")
            return
        if not age_str.isdigit() or int(age_str) <= 0:
            self.status_label.configure(text="Error: Age must be a positive number.", text_color="red")
            return
        
        success, message = db.add_user(name, int(age_str))
        color = "green" if success else "red"
        self.status_label.configure(text=message, text_color=color)
        if success:
            self.name_entry.delete(0, 'end')
            self.age_entry.delete(0, 'end')
            self.refresh_user_list()

    def remove_user_action(self, user_id):
        success, message = db.remove_user(user_id)
        color = "green" if success else "red"
        self.status_label.configure(text=message, text_color=color)
        if success:
            self.refresh_user_list()

    def refresh_user_list(self):
        for widget in self.user_list_frame.winfo_children():
            widget.destroy()
        all_users = db.get_all_users()
        for user in all_users:
            row_frame = ctk.CTkFrame(self.user_list_frame)
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure(0, weight=1)
            label_text = f"ID: {user['id']} | {user['username']} (Age: {user['age']})"
            ctk.CTkLabel(row_frame, text=label_text).grid(row=0, column=0, padx=5, sticky="w")
            if user['role'] != 'admin':
                remove_button = ctk.CTkButton(
                    row_frame, text="Remove", width=80, fg_color="red", hover_color="#c00",
                    command=lambda u_id=user['id']: self.remove_user_action(u_id)
                )
                remove_button.grid(row=0, column=1, padx=5)

class MainAppFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.app = master

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar (Unchanged) ---
        sidebar_frame = ctk.CTkFrame(self, width=150, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsw")
        ctk.CTkLabel(sidebar_frame, text="Controls", font=("Roboto", 24, "bold")).pack(padx=20, pady=(20, 10))
        self.feedback_button = ctk.CTkButton(sidebar_frame, text="View Feedback", command=lambda: self.show_screen("feedback_screen"))
        self.feedback_button.pack(padx=20, pady=10, fill="x")
        self.audio_status_label = ctk.CTkLabel(sidebar_frame, text="Status: Ready", wraplength=130, font=("Roboto", 16))
        self.audio_status_label.pack(padx=20, pady=20)
        self.speaking_indicator = ctk.CTkProgressBar(
            sidebar_frame,
            mode="indeterminate",
            height=4  # A nice, thin bar
        )
        self.transcript_label = ctk.CTkLabel(sidebar_frame, text="You said: ...", wraplength=130, font=("Roboto", 14, "italic"), anchor="w")
        self.transcript_label.pack(padx=20, pady=(40, 0), fill="x")

        # --- Main Container (Unchanged) ---
        main_container = ctk.CTkFrame(self, corner_radius=10)
        main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        # --- Interview Screen (Unchanged) ---
        self.interview_screen_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        self.interview_screen_frame.grid_rowconfigure(1, weight=1)
        self.interview_screen_frame.grid_columnconfigure((0, 1), weight=1)
        self.background_button = ctk.CTkButton(self.interview_screen_frame, text="Start Background Interview", command=lambda: self.app.start_interview_session("Background"))
        self.background_button.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")
        self.salary_button = ctk.CTkButton(self.interview_screen_frame, text="Start Salary Negotiation", command=lambda: self.app.start_interview_session("Salary Negotiation"))
        self.salary_button.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="ew")
        self.chat_history_frame = ctk.CTkScrollableFrame(self.interview_screen_frame, label_text="Interview Transcript")
        self.chat_history_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.chat_history_frame.grid_columnconfigure(0, weight=1)

        # =============================================================== #
        # ---            THIS IS THE CORRECTED UI SECTION             --- #
        # =============================================================== #
        self.feedback_screen_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        self.feedback_screen_frame.grid_rowconfigure(1, weight=1)
        self.feedback_screen_frame.grid_columnconfigure(1, weight=1)

        # --- NEW: A top frame to hold the buttons and title ---
        feedback_top_frame = ctk.CTkFrame(self.feedback_screen_frame, fg_color="transparent")
        feedback_top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        feedback_top_frame.grid_columnconfigure(2, weight=1) # Allow title to expand

        # --- NEW: The "Back to Interview" button ---
        self.return_button = ctk.CTkButton(feedback_top_frame, text="< Back to Interview", command=lambda: self.show_screen("interview_screen"))
        self.return_button.grid(row=0, column=0, padx=(0, 10), sticky="w")
        
        # --- THIS IS THE MISSING BUTTON ---
        self.discuss_button = ctk.CTkButton(
            feedback_top_frame, text="Start Audio Feedback Session",
            command=self.app.start_feedback_session,
            state="disabled" # Starts disabled
        )
        self.discuss_button.grid(row=0, column=1, padx=10, sticky="w")
        # ------------------------------------

        ctk.CTkLabel(feedback_top_frame, text="Feedback Reports", font=("Roboto", 24, "bold")).grid(row=0, column=2, sticky="w")

        # --- The rest of the feedback screen widgets ---
        self.interview_list_frame = ctk.CTkScrollableFrame(self.feedback_screen_frame, label_text="Past Sessions", width=250)
        self.interview_list_frame.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="ns")
        self.report_display_textbox = ctk.CTkTextbox(self.feedback_screen_frame, wrap="word", font=("Roboto", 14), state="disabled")
        self.report_display_textbox.grid(row=1, column=1, padx=(5, 10), pady=(0, 10), sticky="nsew")
    
    def show_screen(self, screen_name):
        """
        Hides all screens, shows the selected one, and correctly manages the
        application's listener state for different modes.
        """
        # First, hide all main content frames
        self.interview_screen_frame.grid_forget()
        self.feedback_screen_frame.grid_forget()

        # Now, decide which screen to show and what actions to take
        if screen_name == "interview_screen":
            # If we are in feedback mode, we must exit it before showing the interview screen.
            self.app.exit_feedback_mode_if_active()
            
            # Show the interview screen and play its specific prompt
            self.interview_screen_frame.grid(row=0, column=0, sticky="nsew")
            self.app.play_audio("interview_screen_prompt")

        elif screen_name == "feedback_screen":
            # When entering the feedback screen, we must start the dedicated feedback listener.
            
            # Show the feedback screen frame
            self.feedback_screen_frame.grid(row=0, column=0, sticky="nsew")
            
            # Silently populate the visual list of past sessions in the UI
            self.app.populate_interview_list()
            
            # This is the crucial call that starts the entire voice-driven feedback flow.
            # The listener itself will handle all audio prompts from this point on.
            self.app.enter_feedback_mode()