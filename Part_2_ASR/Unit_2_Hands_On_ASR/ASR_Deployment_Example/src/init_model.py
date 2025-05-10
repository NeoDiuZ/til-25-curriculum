import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def save_whisper_model_and_processor(
    model_name="openai/whisper-small", output_dir="app/models"
):
    """
    Downloads and saves the Whisper model and processor locally.

    Args:
        model_name (str): The name of the Whisper model to download.
                          Options: "openai/whisper-tiny", "openai/whisper-base",
                          "openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v2"
        output_dir (str): Directory to save the model and processor.
    """
    print(f"Downloading {model_name} model and processor...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create specific directories for model and processor
    model_dir = os.path.join(output_dir, "whisper_model")
    processor_dir = os.path.join(output_dir, "whisper_processor")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(processor_dir, exist_ok=True)

    # Download and save the model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    print(f"Saving model to {model_dir}")
    model.save_pretrained(model_dir)

    print(f"Saving processor to {processor_dir}")
    processor.save_pretrained(processor_dir)

    print(f"Model and processor saved successfully to {output_dir}")

    # Return paths for verification
    return {"model_path": model_dir, "processor_path": processor_dir}


def verify_saved_model(model_path, processor_path):
    """
    Verifies that the model and processor can be loaded from the saved paths.

    Args:
        model_path (str): Path to the saved model.
        processor_path (str): Path to the saved processor.

    Returns:
        bool: True if model and processor load successfully.
    """
    try:
        print("Verifying model can be loaded...")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

        print("Verifying processor can be loaded...")
        processor = WhisperProcessor.from_pretrained(processor_path)

        print(
            "Verification successful! Model and processor can be loaded from saved paths."
        )
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    # You can change the model size here if needed
    model_name = "openai/whisper-small"
    output_dir = "models"

    # Save the model and processor
    paths = save_whisper_model_and_processor(model_name, output_dir)

    # Verify the saved model and processor
    verify_saved_model(paths["model_path"], paths["processor_path"])
