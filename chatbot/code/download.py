import requests
import os

url = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin'
output_filename = './model.bin' 

if not os.path.exists(output_filename):
    response = requests.get(url, stream=True)

    with open(output_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):  
            file.write(chunk)

    print(f"File downloaded as '{output_filename}'")

