import gradio as gr
from chatbot_logic import qa_chain_with_memory

def add_text(history, text):
  # Adding user query to the chatbot and chain
  # use history with curent user question
  if not text:
      raise gr.Error('Enter text')
  history = history + [(text, '')]
  return history

def generate_bot_response(history,query):
  """Function takes the query, history and inputs from the qa chain when the submit button is clicked
  to generate a response to the query"""


  bot_response = qa_chain_with_memory({"query": query})


  # simulate streaming
  for char in bot_response['result']:
          history[-1][-1] += char
          time.sleep(0.05)
          yield history,''


# The GRADIO Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Row():
            # Chatbot interface
            chatbot = gr.Chatbot(label="DeciLM-7B-instruct bot",
                                value=[],
                                elem_id='chatbot')

    with gr.Column():
        with gr.Column():
            # Ask question input field
            txt = gr.Text(show_label=False, placeholder="Enter question")

        with gr.Column():
            # Button to submit question to the bot
            submit_btn = gr.Button('Ask')

    # Event handler for submitting text question and generating response
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=generate_bot_response,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt]
    )

if __name__ == "__main__":
    demo.launch()  # Launch the Gradio app
