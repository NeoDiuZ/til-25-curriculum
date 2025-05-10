# 🎙️ ASR Deployment Tutorial

<div align="center">

![Whisper](https://img.shields.io/badge/Whisper-OpenAI-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Powered-2496ED?logo=docker)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)

</div>

A step-by-step tutorial demonstrating how to deploy an Automatic Speech Recognition (ASR) system using FastAPI and Docker.

## ✨ Overview

The system leverages **Whisper** - OpenAI's powerful pre-trained ASR model - to accurately transcribe audio files. This tutorial walks you through creating a RESTful API that accepts base64-encoded audio and returns transcriptions, ideal for learning ASR deployment concepts.

## 🔍 Prerequisites

- ⚡ Python 3.10+
- 🐳 Docker
- 🖥️ GPU (optional, but recommended for faster inference)

## 🚀 Deployment Instructions

### 1️⃣ Install Requirements

Install the required Python packages with a single command:

```bash
pip install -r requirements.txt
```

### 2️⃣ Initialize the Model

If the `models` folder in the `src` directory is not found or empty, you have two stylish options:

- 📁 Add your own Whisper model to the `src/models` directory
- ⚙️ Run the initialization script to download a pre-trained model:

```bash
python src/init_model.py
```

This will download the Whisper-small model and save it to the `models` directory.

### 3️⃣ Build and Run with Docker

Build a Docker image:

```bash
docker build -t stt_app:1.0.0 .
```

Launch your container:

```bash
docker run -p 8000:8000 stt_app
```

This will start the ASR service on port 8000, ready to transform speech to text.

### 4️⃣ Test the Server

Verify your setup with the provided test script:

```bash
python test.py
```

This will send a test audio file (`audio.mp3`) to the server and display the transcription result.

## 📡 API Usage

The API accepts POST requests to the `/stt` endpoint with a JSON payload containing base64-encoded audio:

```json
{
  "instances": [
    {
      "b64": "base64_encoded_audio_content"
    }
  ]
}
```

The response will be delivered in this format:

```json
{
  "predictions": [
    "transcribed text goes here"
  ]
}
```

