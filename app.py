import gradio as gr
from similarity import get_answer

def chat(query):
    """
    Function called by Gradio to answer user questions.
    """
    return get_answer(query)

# Create Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs="text",
    title="FAQ Chatbot",
    description="Type a question and get the best matching answer from FAQs."
)

if __name__ == "__main__":
    # Launch the Gradio app
    # share=True creates a temporary public link (optional)
    iface.launch(share=False)