import requests
import base64

# The endpoint URL
url = "http://localhost:8000/stt"

# Path to an audio file
audio_file_path = "audio.mp3"  # Replace with your actual file path

# Read the file and encode it to base64
with open(audio_file_path, "rb") as audio_file:
    audio_content = audio_file.read()
    base64_encoded = base64.b64encode(audio_content).decode("utf-8")

# Create payload with base64 encoded audio
data = {"instances": [{"b64": base64_encoded}]}

# Sending the POST request with JSON data
response = requests.post(url, json=data)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response:", response.json())
