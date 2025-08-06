
# Pragati AI: A Bridge from Potential to Progress 🚀

## Overview

**Pragati AI** (Hindi for "progress") is a fully self-contained, offline-first desktop application designed to provide **realistic, voice-driven AI interview coaching**. It's specifically built for deployment in shared educational environments like schools, colleges, and NGOs, aiming to bridge the gap between education and employment for visually impaired and marginalized students.

Our solution focuses on creating a safe, private, and repeatable practice space where users can build confidence and refine their interview skills at their own pace, without needing an internet connection. After each session, the application provides detailed, actionable feedback on both vocal delivery and content.

## Key Features

Pragati AI offers a comprehensive coaching experience through its integrated features:

-   **Dual Interview Modules**:
    
    -   **In-depth Background Interviews**: An AI intelligently asks follow-up questions about a user's projects, experience, and behavioral skills.
        
    -   **Realistic Salary Negotiation**: A dedicated HR persona simulates the challenging back-and-forth of a salary discussion, teaching users how to advocate for themselves.
        
-   **Comprehensive Feedback Engine**: After each interview, the application performs a multi-faceted analysis, evaluating vocal delivery (e.g., words per minute) and content structure (e.g., STAR method adherence, professionalism). It provides scores and qualitative, AI-generated reasons for improvement.
    
-   **Long-Term Progress Tracking**: All feedback reports are saved to a structured SQLite database. Users can vocally request detailed reports on past performance, including comparisons across multiple interviews to track growth and identify trends.
    
-   **Multi-User Profile Management**: A secure SQLite database manages individual user profiles, ensuring personal conversation histories and feedback reports remain private and organized, even on a shared computer.
    
-   **Voice-First, Accessible Design**: The entire application is designed to be fully navigable and operable by voice. This is complemented by a clean, high-contrast, large-text user interface built with CustomTkinter, ensuring a seamless experience for visually impaired and low-vision users.
    

## Technology Stack 💻

To achieve a high-quality, fully offline experience, Pragati AI uses a powerful stack of local models and libraries:

-   **Core AI Engine (LLM)**: **Gemma 3n 2eb Q2_K (2-bit quantized) GGUF** loaded via `llama-cpp-python`. This provides core conversational and analytical intelligence in a highly efficient, CPU-optimized format, bundled directly into the application.
    
-   **Audio Pipeline (ASR & TTS)**:
    
    -   **Speech-to-Text**: OpenAI's **Whisper** for state-of-the-art transcription accuracy.
        
    -   **Text-to-Speech**: **Piper TTS** for natural, high-quality, and responsive voice synthesis.
        
-   **Application Framework**: **CustomTkinter** for a modern, lightweight, and cross-platform graphical user interface.
    
-   **Data Management**: **SQLite** for robust user profile management and **Pandas** for operations on structured feedback data.
    
-   **Data Validation**: **Pydantic** acts as a gatekeeper for data quality, validating data against a strict schema before saving.
    

## System Architecture

The architecture is designed for simplicity and robustness, consisting of three main parts:

### 1. User Input

All user interaction is **voice-based**, from high-level commands ("start an interview") to detailed answers during a session.

### 2. Pragati AI Desktop App (Self-Contained Executable)

This is the core of the system, a single, self-contained executable file created with PyInstaller, requiring no external dependencies or internet connection. It comprises:

-   **UI & Profile Management**: The CustomTkinter front-end and SQLite database logic.
    
-   **Main Orchestrator**: The central "brain" that listens to commands, decides which mode to activate (Interview, Feedback), and manages application flow.
    
-   **AI Engine**: Gemma for conversation and reasoning, and Whisper for speech-to-text.
    
-   **Feedback Analysis**: Module for calculating performance metrics and generating insights.
    

### 3. Application Outputs

-   **Auditory Output**: Primary, voice-first output using Piper TTS for all AI responses, creating a natural, conversational experience.
    
-   **Data Persistence**: Local storage of user profiles, conversation history, and detailed feedback reports to ensure privacy and enable long-term progress tracking.
    

## Development Journey

The development of Pragati AI followed an iterative process, building from a stable core to a robust, feature-rich application.

### Phase 1: Architecting the Core Application & Proving the Voice Loop

Focused on establishing a stable, real-time "voice-in, voice-out" loop and a robust application architecture. Key decisions included selecting Whisper and Piper for audio, CustomTkinter for UI, and switching to GGUF with `llama-cpp-python` for truly self-contained AI.

### Phase 2: Implementing and Refining the Conversational Flows

Transformed the basic Q&A bot into a stateful, intelligent interviewer. This involved developing `interview_flow_manager.py` to prevent conversational stagnation using Jaccard similarity checks and implementing state-aware prompts for natural progression through interview stages.

### Phase 3: Building the Voice-Driven Feedback System

Underwent a significant redesign from a UI-first approach to a fully voice-based navigation for accessibility. This required creating a dedicated feedback listener to manage complex, contextual conversations for report selection.

### Phase 4: The Final Polish - From Functional to Professional

Focused on refining the user experience, ensuring smooth transitions, clear audio cues, and overall professional presentation.

## Installation & Usage

Pragati AI is designed to be highly portable and operable offline. You have two main ways to use it:

### Option 1: Running from Executable (Offline Portability)

For maximum convenience and offline use, you can run the pre-built executable. This version is entirely self-contained and **does not require an internet connection or any Python installation once transferred**. It's perfect for distributing via a USB drive or local network in environments without reliable internet.

1.  **Download the latest release** from the [Google Drive Link](https://drive.google.com/drive/folders/1MlQh1CuAXGBFS0p3AsagMEGzvFrOscHU?usp=sharing).
    
2.  **Unzip the downloaded file**.
    
3.  **Transfer the entire unzipped folder** to your desired machine (e.g., via a pen drive).
    
4.  **Run `Pragati-AI.exe`** (or the equivalent executable name) from within the transferred folder.
    

### Option 2: Running from Source (Recommended for Best Experience)

For the **best performance and development experience**, we recommend running Pragati AI directly from its Python source code. This allows for easier debugging, customization, and ensures you're running with the most optimized environment.

1.  **Clone the repository**:
    
    ```
    git clone https://github.com/singhamal001/Pragati-AI.git
    cd Pragati-AI
    
    ```
    
2.  **Create a virtual environment** and activate it:
    
    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    ```
    
3.  **Install dependencies**:
    
    ```
    pip install -r requirements.txt
    
    ```
    
4.  **Download the Gemma 3n 2eb Q2_K GGUF model** (e.g., `gemma-3n-e2b-it.Q2_K_M.gguf`) and place it in the `./model/` directory. You can typically find this on Unsloth's repository [https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/tree/main] -> `gemma-3n-E2B-it-Q2_K.gguf` on Hugging Face .
    
5.  **Download the Piper TTS model** (e.g., `en_US-hfc_female-medium.onnx`) and place it in the `./model/` directory. This can be found on the Piper TTS GitHub releases page.
    
6.  **Run the application**:
    
    ```
    python app.py
    
    ```
    

## Future Roadmap

Our vision for Pragati AI extends further:

-   **Data-Driven Mobile App Strategy**: Conduct detailed user research to validate the need for a mobile application, ensuring future platform expansion is driven by genuine user requirements.
    
-   **Adaptive Interviewing Engine**: Implement an advanced AI engine where the interviewer dynamically adjusts question difficulty and focus areas based on the user's real-time performance, creating a truly personalized coaching session.
    

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE "null") - see the `LICENSE` file for details.

## Contact

For any questions or inquiries, please contact [Amal Singh - singhamal1710@gmail.com] || [Manasvi Logani - mlogani2001@gmail.com] .
