import gradio as gr
import os
import io
import requests
import json
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

model_id = os.getenv("model_id")
hf_api_key = os.getenv("hf_api_key")

api_url =f"https://api-inference.huggingface.co/models/{model_id}"

def get_completion(inputs, parameters=None, ENDPOINT_URL=api_url): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']

gr.close_all()

demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="开源文本总结AI Assistant",
                    description="Input the contents you want the AI Assistant to summarize for you."
                   )

demo.launch()
