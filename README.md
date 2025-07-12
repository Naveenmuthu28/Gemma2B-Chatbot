# Gemma2B-Chatbot
This is a local chatbot web application powered by Google's Gemma 2B Instruction-Tuned model. It supports multiple chat sessions with per-session memory, built using Hugging Face Transformers, PyTorch, and Gradio. Runs completely offline on your own machine (CPU or GPU).

---

## Features

- Runs locally with GPU acceleration (CUDA)
- Remembers chat history per session
- Multiple chat sessions (New Chat, Delete Chat, Switch Chat)
- Clean, responsive Gradio web interface
- No API keys or internet required after model download

---

## Getting Started

### 1. Clone the Repository
       ```bash
       git clone https://github.com/yourusername/gemma2b-chatbot.git
       cd gemma2b-chatbot

### 2. Install Requirements
  Make sure you have Python 3.9–3.11 installed.
       ```bash
       pip install -r requirements.txt
       
   Or manually install:
        ```bash
        pip install torch transformers gradio
       
### 3. Create a account and login to Hugging Face
  - search for Gemma 2B model and accept terms.
  - Go to tokens page, create new token in read mode and copy it.
  - go to cmd, to login into hugging face.
      ```bash
      huggingface-cli login
      ```
  - paste the token.

### 4. Run the script
       ```bash
       python main.py
       ```
---

## Project Structure

Gemma2B-Chatbot/
│
├── main.py 
├── requirements.txt
├── README.md
└── LICENSE

---

## Notes

- First run will download the Gemma 2B model (~4GB–5GB).
- Runs fully offline after download.
- Supports both CPU and GPU (recommended: GPU with at least 8GB VRAM).
- Multi-session chat memory is handled in memory per app run.

---

## LICENSE

This project uses open weights from Google's Gemma 2B model under the terms set by Google on Hugging Face.
The code in this repository is provided under a permissive license (e.g., MIT).
