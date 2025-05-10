# 🤖 RL Deployment Example Tutorial

<div align="center">
  
  ![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-Deployment-blue?style=for-the-badge)
  ![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)
  ![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
  
</div>

---

This tutorial guides you through deploying a reinforcement learning model in a production-like environment using Docker and FastAPI. This tutorial uses the [MountainCar-v0 environment](https://gymnasium.farama.org/environments/classic_control/mountain_car/) as an example

## 🚀 Getting Started

### Prerequisites
- Docker installed
- Python 3.10 installed
- Install required packages: `pip install -r requirements.txt`

### 📋 Setup Instructions

**1️⃣ Prepare the model**

If no models exist in the `src` directory, you have two options:
- 🔹 Add your own model and create a `models` folder manually
- 🔹 Run `init_model.py` to generate a sample model:
  ```bash
  python init_model.py
  ```

**2️⃣ Run the Docker container**

In one terminal, execute the following commands:
```bash
sudo docker build -t rl_app:1.0.0 .
sudo docker run -p 8000:8000 rl_app:1.0.0
```

**3️⃣ Test the deployment**

In another terminal, run:
```bash
python test.py
```

## 📊 API Usage

The API endpoint is available at `http://localhost:8000/rl` and accepts POST requests with the following JSON payload format:

```json
{
  "instances": [
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
  ]
}
```

Where each array within "instances" represents an observation/state from the environment.

Response format:

```json
{
  "predictions": [
    {"action": [0]},
    {"action": [1]},
    {"action": [2]}
  ]
}
```

You can also check the health of the API with a GET request to `http://localhost:8000/health`.

## 📊 What to Expect

After successful deployment, your RL model will be accessible via API, allowing for real-time inference and interaction.


