# Voice-Controlled GUI Application

This is a desktop application built with Python that demonstrates how to control a graphical user interface (GUI) using voice commands. The application features a modern, clean UI and uses a "wake word" to activate its command-listening mode.

The primary goal of this project is to provide a foundational example of integrating voice recognition into a desktop application for hands-free navigation.

![App Screenshot](https://i.imgur.com/8a1bJ2M.png)
*(A preview of the application's interface, showing the sidebar with the status label and the main screen area.)*

## Key Features

- **Modern User Interface:** Built with the **CustomTkinter** library for a sleek, modern look and feel.
- **Wake Word Activation:** The application is always listening in the background for the wake word **"Pragati"**. It does not process any other commands until it hears this word.
- **Voice-Controlled Navigation:** After activation, the application listens for a command and can switch between three different screens based on what the user says (e.g., "Go to screen two").
- **Real-time Visual Feedback:** A status label in the sidebar provides constant feedback to the user, indicating whether the app is listening, has heard the wake word, or is processing a command.
- **Non-Blocking Listener:** The voice recognition runs in a separate thread, ensuring the GUI remains responsive at all times.

## Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

- **Python 3.7+**
- A working **microphone**.
- An active **internet connection** (required for the Google Web Speech API).

### Installation

1.  **Clone the repository or download the source code.**
    If you have git installed, you can clone the repository. Otherwise, simply ensure `app.py` is in its own folder.

2.  **Install the required Python libraries.**
    All necessary packages can be installed using pip. Open your terminal or command prompt and run:
    ```sh
    pip install customtkinter SpeechRecognition
    ```
    > **Note:** The `SpeechRecognition` library requires `PyAudio` to access the microphone. It should be installed automatically. If you encounter errors with `PyAudio`, you may need to install it manually.

## How to Use

1.  **Run the application** from your terminal in the project directory:
    ```sh
    python app.py
    ```

2.  Wait for the application window to appear. In the sidebar, the status will update to: **"Status: Listening for 'Pragati'..."**.

3.  Say the wake word clearly into your microphone: **"Pragati"**.

4.  The status label will change to: **"Status: Wake word detected! Listening for command..."**.

5.  Immediately give your command. For example:
    - *"Go to screen two"*
    - *"Show screen 3"*
    - *"Switch to screen 1"*

6.  The application will process your command, switch to the corresponding screen, and update the status. After a moment, it will return to listening for the wake word again.

## Technology Stack

- **CustomTkinter:** Used to create the modern graphical user interface.
- **SpeechRecognition:** A library for performing speech recognition, used here with the Google Web Speech API to convert spoken language into text.
- **PyAudio:** Used under the hood by `SpeechRecognition` to capture audio from the microphone.
- **Threading:** Python's built-in threading module is used to run the voice listener in the background, preventing the GUI from freezing.