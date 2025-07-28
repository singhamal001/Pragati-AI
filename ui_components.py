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
    def __init__(self, master):
        super().__init__(master)
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        sidebar_frame = ctk.CTkFrame(self, width=150, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsw")
        ctk.CTkLabel(sidebar_frame, text="Controls", font=("Roboto", 24, "bold")).pack(padx=20, pady=(20, 10))
        self.audio_status_label = ctk.CTkLabel(sidebar_frame, text="Status: Ready", wraplength=130, font=("Roboto", 16))
        self.audio_status_label.pack(padx=20, pady=20)

        # --- NEW: Transcript Label ---
        self.transcript_label = ctk.CTkLabel(
            sidebar_frame, 
            text="You said: ...", 
            wraplength=130, 
            font=("Roboto", 14, "italic"),
            anchor="w"
        )
        self.transcript_label.pack(padx=20, pady=(40, 0), fill="x")


        main_container = ctk.CTkFrame(self, corner_radius=10)
        main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        self.interview_screen_frame = ctk.CTkFrame(main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.interview_screen_frame, text="Interview Screen", font=("Roboto", 32, "bold")).pack(pady=50)
        
        self.feedback_screen_frame = ctk.CTkFrame(main_container, fg_color="#2a3b47")
        ctk.CTkLabel(self.feedback_screen_frame, text="Feedback Screen", font=("Roboto", 32, "bold")).pack(pady=50)

    def show_screen(self, screen_name):
        """Hides all screens and shows the selected one."""
        self.interview_screen_frame.grid_forget()
        self.feedback_screen_frame.grid_forget()
        
        if screen_name == "interview_screen":
            self.interview_screen_frame.grid(row=0, column=0, sticky="nsew")
        elif screen_name == "feedback_screen":
            self.feedback_screen_frame.grid(row=0, column=0, sticky="nsew")