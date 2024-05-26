import cv2
import numpy as np

from torch import tensor, float32
from fastapi import UploadFile, FastAPI, HTTPException, File
from typing import Union
from ultralytics import YOLO


async def file_to_image(file: Union[UploadFile, bytes]):
    
    # load image from buffer
    if isinstance(file, UploadFile):
        contents = await file.read()
    elif isinstance(file, bytes):
        contents = file
    else:
        raise HTTPException(status_code=400, detail="Invalid file format!")
    
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return tensor(image, dtype=float32)


poseCOCO_model = None


app = FastAPI()


@app.on_event("startup")
def load_models():
    global poseCOCO_model
    poseCOCO_model = YOLO("./PoseCOCO.pt").cuda()


@app.post("/poseCOCO")
async def get_PoseCOCO(file: Union[UploadFile, bytes] = File(...)):
    
    assert poseCOCO_model is not None, "Model not loaded"
    
    image_tensor = file_to_image(file)
    
    ret = poseCOCO_model.predict(image_tensor)
    
    return {
        "status": "success",
        "result": ret,
    }