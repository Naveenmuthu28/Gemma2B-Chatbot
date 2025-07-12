import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model and tokenizer
model_name = "google/gemma-2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# Session management
sessions = [[]]  # Start with one empty session
current_session_index = 0

# Utility: format one session as string prompt
def build_prompt(session):
    prompt = "\n".join(session)
    if session and session[-1].startswith("User:"):
        prompt += "\nBot:"
    return prompt

# Core chat logic
def chat(user_input):
    global sessions, current_session_index

    session = sessions[current_session_index]
    session.append(f"User: {user_input}")

    prompt = build_prompt(session)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only bot's reply
    if "Bot:" in output_text:
        reply = output_text.split("Bot:")[-1].strip()
    else:
        reply = output_text.strip()

    session.append(f"Bot: {reply}")
    return reply, format_display(session)

# Format session for chatbot display
def format_display(session):
    display = []
    for i in range(1, len(session), 2):
        user_msg = session[i - 1].replace("User: ", "") if "User:" in session[i - 1] else ""
        bot_msg = session[i].replace("Bot: ", "") if "Bot:" in session[i] else ""
        display.append((user_msg, bot_msg))
    return display

# New Chat
def new_chat():
    global sessions, current_session_index
    sessions.append([])
    current_session_index = len(sessions) - 1
    return gr.update(choices=[f"Chat {i+1}" for i in range(len(sessions))], value=f"Chat {current_session_index + 1}"), []

# Delete Chat
def delete_chat():
    global sessions, current_session_index

    if len(sessions) <= 1:
        sessions = [[]]
        current_session_index = 0
    else:
        sessions.pop(current_session_index)
        if current_session_index >= len(sessions):
            current_session_index = len(sessions) - 1

    choices = [f"Chat {i+1}" for i in range(len(sessions))]
    value = f"Chat {current_session_index + 1}"
    display = format_display(sessions[current_session_index])

    return gr.update(choices=choices, value=value), display

# Switch Chat
def switch_chat(choice):
    global sessions, current_session_index
    index = int(choice.split()[-1]) - 1
    current_session_index = index
    return format_display(sessions[index])

# Custom CSS
css = """
body, html {
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}

.gradio-container, .gradio-container > * {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

.row, .column {
    margin: 0 !important;
    padding: 0 !important;
}
"""

# Gradio UI
with gr.Blocks(theme="soft", css=css) as demo:
    with gr.Row():
        with gr.Column(scale=0.6, min_width=200):
            gr.Markdown("Your Chats")
            chat_selector = gr.Dropdown(choices=["Chat 1"], value="Chat 1", label="Select a chat")
            new_chat_btn = gr.Button("\u2795 New Chat")
            delete_chat_btn = gr.Button("\U0001F5D1\ufe0f Delete Chat")

        with gr.Column(scale=4.4):
            gr.Markdown("Gemma-2B Chatbot with Memory")
            chatbot = gr.Chatbot(height=700)
            msg = gr.Textbox(placeholder="Type your message here and press Enter")

    # Interactions
    def respond(user_message, chat_display):
        reply, new_display = chat(user_message)
        return "", new_display

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    new_chat_btn.click(new_chat, outputs=[chat_selector, chatbot])
    delete_chat_btn.click(delete_chat, outputs=[chat_selector, chatbot])
    chat_selector.change(switch_chat, inputs=chat_selector, outputs=chatbot)

# Launch
demo.launch(inbrowser=True)