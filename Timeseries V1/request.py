import requests
import json

url = 'http://127.0.0.1:5000/predict_api'

data = {'Memory_Usage':48}

r = requests.post(url,json=data)

print("Model prediction :", r.json())