FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

WORKDIR /opt/app

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    python3.10 \
    pip \
    wget

# install ultralytics
RUN pip install ultralytics
RUN pip install \
    fastapi \
    uvicorn

# load models
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt -O ./DetectionCOCO.pt
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-oiv7.pt -O ./DetectionOpenImagev7.pt
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt -O ./SegmentationCOCO.pt
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-pose-p6.pt -O ./PoseCOCO.pt
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-obb.pt -O ./OBBDOTAv1.pt
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-cls.pt -O ./ClassificationImageNet.pt



COPY . .
CMD [ "uvicorn", "routs:app", "--host", "0.0.0.0", "--port", "8080" ]