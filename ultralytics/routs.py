import cv2
import numpy as np

from torch import tensor, float32
from torchvision.transforms import Resize
from fastapi import UploadFile, FastAPI, HTTPException, File
from typing import Union
from ultralytics import YOLO


async def file_to_image(file: Union[UploadFile, bytes], out_size=(640, 640)):
    
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
    
    image = tensor(image, dtype=float32).cuda() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = Resize(out_size)(image)
    
    return image

# MODELS -----------------------------------------------------

detectionCOCO_model = None
poseCOCO_model = None
classificationImageNet_model = None
detectionOpenImagev7_model = None
segmentationCOCO_model = None
obbDOTAv1_model = None

app = FastAPI()


@app.on_event("startup")
def load_models():
    global detectionCOCO_model
    global poseCOCO_model
    global classificationImageNet_model
    global detectionOpenImagev7_model
    global segmentationCOCO_model
    global obbDOTAv1_model
    
    obbDOTAv1_model = YOLO("./OBBDOTAv1.pt").cuda()
    detectionCOCO_model = YOLO("./DetectionCOCO.pt").cuda()
    poseCOCO_model = YOLO("./PoseCOCO.pt").cuda()
    classificationImageNet_model = YOLO("./ClassificationImageNet.pt").cuda()
    detectionOpenImagev7_model = YOLO("./DetectionOpenImagev7.pt").cuda()
    segmentationCOCO_model = YOLO("./SegmentationCOCO.pt").cuda()


@app.post("/detectionCOCO")
async def detectionCOCO(file: Union[UploadFile, bytes] = File(...)):
    
    assert detectionCOCO_model is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = detectionCOCO_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        results.append(
            {
                "classes": ret[i].names,
                "bboxes": [{
                    "bbox": box.xywhn.cpu().numpy().tolist()[0],
                    "confidence": box.conf.cpu().numpy().tolist()[0],
                    "class": int(box.cls.cpu().numpy().tolist()[0])
                } for box in ret[i].boxes]
            }
        )
    
    return {
        "status": "success",
        "result": results
    }


@app.post("/poseCOCO")
async def poseCOCO(file: Union[UploadFile, bytes] = File(...)):
    
    assert poseCOCO_model is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = poseCOCO_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        results.append({
            "bboxes": [{
                    "bbox": box.xywhn.cpu().numpy().tolist()[0],
                    "confidence": box.conf.cpu().numpy().tolist()[0],
                    "class": int(box.cls.cpu().numpy().tolist()[0])
                } for box in ret[i].boxes],
            "landmarks": [{
                    "keypoints": lan.xyn.cpu().numpy().tolist(),
                    "confidence": lan.conf.cpu().numpy().tolist()
                } for lan in ret[i].keypoints]
        })
    
    return {
        "status": "success",
        "result": results
    }

@app.post("/claaificationImageNet")
async def claaificationImageNet(file: Union[UploadFile, bytes] = File(...)):
    
    assert classificationImageNet_model  is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = classificationImageNet_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        results.append({
            "classes": ret[i].names,
            "probability": ret[i].probs.data.cpu().numpy().tolist(),
            "Top5": map(int, ret[i].probs.top5)
        })
    
    return {
        "status": "success",
        "result": results
    }

@app.post("/detectionOpenImagev7")
async def detectionOpenImagev7(file: Union[UploadFile, bytes] = File(...)):
    
    assert detectionOpenImagev7_model is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = detectionOpenImagev7_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        results.append({
            "classes": ret[i].names,
            "bboxes": [{
                "bbox": box.xywhn.cpu().numpy().tolist()[0],
                "confidence": box.conf.cpu().numpy().tolist()[0],
                "class": int(box.cls.cpu().numpy().tolist()[0])
            } for box in ret[i].boxes],
        })
    
    return {
        "status": "success",
        "result": results
    }

@app.post("/segmentationCOCO")
async def segmentationCOCO(file: Union[UploadFile, bytes] = File(...)):
    
    assert segmentationCOCO_model is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = segmentationCOCO_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        results.append({
            "classes": ret[i].names,
            "masks": [{
                "class": int(bbox.cls.cpu().numpy().tolist()[0]),
                "confidence": bbox.conf.cpu().numpy().tolist()[0],
                "bbox": bbox.xywhn.cpu().numpy().tolist()[0],
                "mask": mask.xyn[0].tolist()
            } for (bbox, mask) in zip(ret[i].boxes, ret[i].masks)
        ]})
    
    return {
        "status": "success",
        "result": results
    }

@app.post("/obbDOTAv1")
async def obbDOTAv1(file: Union[UploadFile, bytes] = File(...)):
    
    assert obbDOTAv1_model is not None, "Model not loaded"
    
    image_tensor = await file_to_image(file)
    
    ret = obbDOTAv1_model.predict(image_tensor)
    
    results = []
    for i in range(len(ret)):
        
        if ret[i] is None:
            continue
        
        bboxes = [obb.xywhr.cpu().numpy().tolist()[0] for obb in ret[i].obb]
        
        results.append({
            "classes": ret[i].names,
            "bboxes": [{
                "class": int(obb.cls.cpu().numpy().tolist()[0]),
                "confidence": obb.conf.cpu().numpy().tolist()[0],
                "bbox": [
                    bboxes[j][0] / image_tensor.shape[2], 
                    bboxes[j][1] / image_tensor.shape[3], 
                    bboxes[j][2] / image_tensor.shape[2], 
                    bboxes[j][3] / image_tensor.shape[3], 
                    bboxes[j][4]
                ]
            } for (j, obb) in enumerate(ret[i].obb)
        ]})
    
    return {
        "status": "success",
        "result": results
    }