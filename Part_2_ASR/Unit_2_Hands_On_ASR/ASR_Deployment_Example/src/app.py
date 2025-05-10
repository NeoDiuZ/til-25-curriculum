from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import os
import numpy as np
import librosa
import base64
import io
import soundfile as sf

app = FastAPI()

# Fetch the model directory from the environment variable
model_directory = "src/models"
whisper_directory = os.path.join(model_directory, "whisper_model")
processor_directory = os.path.join(model_directory, "whisper_processor")

# Check if we have a fine-tuned model
if os.path.exists(os.path.join(model_directory, "whisper_model", "model.safetensors")):
    # Load fine-tuned model
    processor = WhisperProcessor.from_pretrained(processor_directory)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

asr = pipeline("automatic-speech-recognition", model = model, tokenizer = processor.tokenizer, feature_extractor = processor.feature_extractor, device = device)

@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the file path of an audio file
    Returns transcription of the audio
    """
    input_json = await request.json()

    predictions = []
    for instance in input_json["instances"]:
        audio_bytes = base64.b64decode(instance["b64"])
        audio_np, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        # Save the audio to a file
        audio_file_path = "audio.wav"
        sf.write(audio_file_path, audio_np, 16000)
        fmt_input = {'raw': audio_np, "sampling_rate": 16000}
        result = asr(fmt_input)
        
        transcription = result['text']
        predictions.append(transcription)
    return {"predictions": predictions}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
