import requests
from base64 import b64encode

url = "http://localhost:8000/detect-object"

with open('cat.jpg', 'rb') as f:
    img = f.read()

img_base64 = b64encode(img).decode()

response = requests.post(url, json={"img_base64": img_base64})

print(response.json())
# [{'box': [1022, 1259, 1276, 1997], 'class': 'cat'}]
