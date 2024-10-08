
import json
import os
from threading import Thread
import queue

from openai import AssistantEventHandler, OpenAI
import gradio as gr
from gradio_pdf import PDF
import pandas as pd
from typing_extensions import override

from pandasai import Agent, SmartDataframe
from pandasai.llm.openai import OpenAI as pd_OpenAI

from registered_functions import (
    update_weather,
    update_weather_forecast,
    REGISTERED_TOOLS,
)
from synthetic_data import get_synthetic_data

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]
ASSISTANT_ID = os.environ["ASSISTANT_ID"]
CLIENT = OpenAI()
THREAD = CLIENT.beta.threads.create()
ASSISTANT = CLIENT.beta.assistants.create(
  instructions="You are a weather bot. Use the provided functions to answer questions.",
  model="gpt-4o",
  tools=REGISTERED_TOOLS
)

response_queue = queue.Queue() 

# Define the EventHandler class
class EventHandler(AssistantEventHandler):
    def __init__(self, response_queue):
        super().__init__()
        self.response_queue = response_queue

    @override
    def on_text_created(self, text) -> None:
        pass

    @override
    def on_text_delta(self, delta, snapshot):
        text = delta.value
        self.response_queue.put(text)

    @override
    def on_event(self, event):
      # Retrieve events that are denoted with 'requires_action'
      # since these will have our tool_calls
      if event.event == 'thread.run.requires_action':
        run_id = event.data.id  # Retrieve the run ID from the event data
        self.handle_requires_action(event.data, run_id)
 
    def handle_requires_action(self, data, run_id):
      tool_outputs = []
        
      for tool in data.required_action.submit_tool_outputs.tool_calls:
        if tool.function.name == "update_weather":
            args = json.loads(tool.function.arguments)
            loc = args["location"]
            tool_outputs.append({"tool_call_id": tool.id, "output": get_current_weather(loc)})
        elif tool.function.name == "update_weather_forecast":
            args = json.loads(tool.function.arguments)
            loc = args["location"]
            tool_outputs.append({"tool_call_id": tool.id, "output": get_3_hour_forecast(loc)})
        
      # Submit all tool_outputs at the same time
      self.submit_tool_outputs(tool_outputs, run_id)
 
    def submit_tool_outputs(self, tool_outputs, run_id):
      # Use the submit_tool_outputs_stream helper
      with CLIENT.beta.threads.runs.submit_tool_outputs_stream(
        thread_id=self.current_run.thread_id,
        run_id=self.current_run.id,
        tool_outputs=tool_outputs,
        event_handler=EventHandler(response_queue),
      ) as stream:
        for text in stream.text_deltas:
          print(text, end="", flush=True)
        print()

def chat(usr_message, history):
    """ """
    # start_conversation()
    user_input = usr_message
    global response_queue

    if not THREAD.id:
        print("Error: Missing thread_id")
        return json.dumps({"error": "Missing thread_id"}), 400

    print(
        f"Received message: {user_input} for thread ID: {THREAD.id}"
    )

    # Add the user's message to the thread
    CLIENT.beta.threads.messages.create(
        thread_id=THREAD.id, role="user", content=user_input
    )

    # Create a queue to hold the assistant's response chunks
    global response_queue # = queue.Queue()
    # Start the streaming run in a separate thread
    def run_stream():
        with CLIENT.beta.threads.runs.stream(
            thread_id=THREAD.id,
            assistant_id=ASSISTANT.id,
            event_handler=EventHandler(response_queue)
            ) as stream:
            stream.until_done()


    stream_thread = Thread(target=run_stream)
    stream_thread.start()

    assistant_response = ""
    while True:
        try:
            # Get response chunks from the queue
            chunk = response_queue.get(timeout=0.1)
            assistant_response += chunk
            yield assistant_response
        except queue.Empty:
            # Check if the stream has finished
            if not stream_thread.is_alive():
                break

    # Wait for the stream thread to finish
    stream_thread.join()


llmmodel = pd_OpenAI(api_token=os.environ["OPENAI_API_KEY"], model="gpt-4o")

# Load dataframes
dfcleaned = pd.read_csv("dfcleaned.csv")
dfshaps = pd.read_csv("shaps.csv")
# Initialize Agent
agent = Agent([dfcleaned, dfshaps], config={"llm": llmmodel})
sdfshaps = SmartDataframe(dfshaps, config={"llm": llmmodel})
sdfcleaned = SmartDataframe(dfcleaned, config={"llm": llmmodel})


def process_query(query):
    """ """
    response = agent.chat(query)  # or agent chat, gr.Image
    print(response)
    if isinstance(response, str) and ".png" in response:
        return response, response, None
    elif isinstance(response, str) and ".png" not in response:
        return response, None, None
    elif isinstance(response, pd.DataFrame):
        return None, None, response


def gradio_app():
    """ """
    iface = gr.Interface(
        fn=process_query,
        inputs="text",
        outputs=[
            gr.Textbox(label="Response"),
            gr.Image(label="Plot"),
            gr.DataFrame(label="Dataframe"),
        ],
        title="pandasai Query Processor",
        description="Enter your query related to the csv data files.",
    )
    return iface


with gr.Blocks(
    theme=gr.themes.Monochrome(primary_hue="green"),
) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# Building Automation Assistant")

            gr.Markdown(
                "I'm an AI assistant that can help with building maintenance and equipment questions."
            )

            gr.Markdown("---")

            # Update the ChatInterface to handle streaming
            chat_interface = gr.ChatInterface(
                chat,
                chatbot=gr.Chatbot(
                    height=750,
                    avatar_images=("user_avatar.png", "assistant_avatar.png"),
                ),
                title="Ask Me Anything",
                theme="glass",
                description="Type your question about building automation here.",
                examples=[
                    "Tell me about the HouseZero dataset. Retrieve information from the publication you have access to. Use your file retrieval tool.",
                    "Describe in detail the relationshp between the columns in the two uploaded CSV files and the information you have access to regarding the HouseZero dataset. Be verbose. Use your file retrieval tool.",
                    "Tell be in great detail any advice you have to maintain a small to midsize office building, like the HouseZero data corresponds to. Be verbose. Use your file retrieval tool.",
                    "Tell me in great detail any advice you have for the building managers of large hospitals. Be verbose. Use your file retrieval tool.",
                    "Show massachusetts electricity billing rates during the same time span as the CSV data",
                    "Use those rates and the relevant columns in the CSV files to estimate how much it costs to operate this building per month.",
                    "What is the estimated average electricity cost for operating the building using massachusetts energy rates. use your file retrieval tool. use data csv files for building data. Limit your response to 140 characters. Use your file retrieval tool.",
                    "The anomaly_score field on one of the CSVs indicates that that row is an anomaly if it has value greater than zero.. can you please list a few of the rows with the highest value for this column and using your building efficiency knowledge explain why they may represent a problem? Use your file retrieval tool.",
                    "Based on the data in these CSV files, can you assign an EnergyIQ score from 1-10 that reflects how well the building is operating? Explain the reason for your score and provide any recommendations on actions to take that can improve it in the future. Be verbose. Use your file retrieval tool.",
                    "What would be a good feature to plot as a function of time to illustrate the problems of why the EnergyIQ score is low? Use your file retreival tool.",
                ],
                fill_height=True,
            )

            gr.Markdown("---")

    synthetic_dataset = get_synthetic_data()
    with gr.Row():
        upper_plot = gr.LinePlot(
            synthetic_dataset,
            x="time",
            y="Upper_Thermostat",
            title="Upper Thermostat Temperature",
        )
        lower_plot = gr.ScatterPlot(
            synthetic_dataset,
            x="time",
            y="Lower_Thermostat",
            title="Lower Thermostat Temperature",
        )

        gr.Column([upper_plot, lower_plot], scale=1)

        anomaly_info = gr.Markdown("Anomaly detected around October 15, 2023")
    # Weather input
    with gr.Row():
        iface = gradio_app()


demo.launch(share=True)