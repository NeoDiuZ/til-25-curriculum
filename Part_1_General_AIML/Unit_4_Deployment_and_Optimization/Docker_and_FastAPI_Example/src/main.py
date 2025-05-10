import base64
from io import BytesIO
from typing import Union

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

# Load a pretrained YOLO11n model, pretrained model can be downloaded from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
model = YOLO("models/yolo11n.pt")
conf_threshold = 0.5


class ObjectDetectionRequest(BaseModel):
    img_base64: str


# Receive base64-encoded images and return the object detection results
@app.post("/detect-object")
def detect_object(request: ObjectDetectionRequest) -> list[dict[str, Union[list[int], str]]]:
    # get base64 from request
    img_base64 = request.img_base64
    # convert base64 image to bytes
    img = base64.b64decode(img_base64)
    # read bytes into PIL Image
    img = Image.open(BytesIO(img))
    # perform prediction
    results = model(img)[0]
    class_id_mapping = results.names

    # get the bbox and class
    result_dict = []
    for box, conf, cls in zip(results.boxes.xywh, results.boxes.conf, results.boxes.cls):
        if conf >= conf_threshold:  # filter out predictions below conf_threshold
            result_dict.append(
                # .cpu() first moves tensor from GPU to CPU. If you are running the model on a CPU already then it is not needed.
                # numpy() will convert pytorch tensors into numpy arrays, which can then be converted to Python objects so that they can be returned as JSON to the requester.
                # we need to round the boxes because they are given in float but in real life only integers make sense as you cannot divide a pixel into smaller units
                {'box': map(round, list(box.cpu().numpy())), 'class': class_id_mapping[int(cls.cpu().numpy())]}
            )
    return result_dict
