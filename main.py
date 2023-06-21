from flask import Flask, request
from PIL import Image
import base64
import io
import torch
import cv2
import time
import numpy as np
import os
classes = ['Construction and demolition waste', 'Damaged car', 'streets digging', 'Concrete barriers',
           'Buildings under construction', 'dilapidated sidewalks', 'sanitation', 'water leak', 'trash can',
           'Air conditioning', 'writing on walls','light poles','storage outside the house wall','Road Signs','Outdoor umbrellas']




app = Flask(__name__)
@app.route('/predict', methods=['GET','POST'])

def predict():

    start=time.time()
    imagestr=request.get_data(as_text=True)
    image_data = base64.b64decode(imagestr)
    image = Image.open(io.BytesIO(image_data))
    image.save("my-image.png")
    model = torch.hub.load('yolov5-master/', 'custom', path='bests.pt', source='local')
    image=cv2.imread('my-image.png')
    image=cv2.resize(image,(640,640), interpolation=cv2.INTER_NEAREST)
    output = model(image)
    output=output.xyxy[0]#.to_json(orient="records")
    if len(output) > 0 :
        output=list(map(float, list(output[0])))
        output[-1]=classes[int(output[-1])]
        print(output )
    else:
        end = time.time()
        duration = end - start
        print("duration :", duration)
        return("no detections")
    end=time.time()
    duration=end-start
    print("duration :",duration)
    return (str(output))


if __name__ == '__main__':
    app.run(port=5000)