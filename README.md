# License plate recognition system


This model is the final project of the Data Science Boot Camp at the WBS coding school, Berlin.
The main output here is the streamlit application avaliable at this [link](https://dimarfin-license-plate-recognition-main-streamlit-96fn6k.streamlitapp.com/)

## Model implementation
This model implements detection the car license plate detection and recognition of the text on it.
It includes two main steps several steps depicted in the figure below:
![Figure 1](https://github.com/Dimarfin/License_plate_recognition/docs/fig01.png)

## Object detection step
The detection step is performed using [YOLOv5](https://github.com/ultralytics/yolov5) – a state of the art object detection system. It is based on a constitutional neuronal network and build using PyTorch library. YOLOv5 is supplied pertained on MS COCO dataset which speeds up the training step for a custom object. One of the review of the system can be found at the following [link] (https://towardsdatascience.com/yolo-v5-is-here-b668ce2a4908). 
While there are several sizes of the YOLOv5 in this work YOLOv5s was utilized. For detection of the car license plates it was trained on a [data set from Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).  
The model reached outstanding performance after 20 epochs of training. Some of the performance metrics measured on the validation subset are presented in the figure below.
![Figure 2](https://github.com/Dimarfin/License_plate_recognition/docs/fig02.png)


## Image processing step
Optical character recognition can be improved if images of the detected licence plates are preprocessed to remove noise and unwanted feature which can be misrecognized as a character. This preprocessing step includes several operation such as converting image to a binary form, contour detection, erosion, dilation, etc. After that, prepared image is supplied to the OCR engine.

## Optical character recognition
To recognized text on the license plate tesseract [Tesseract Open Source OCR Engine](https://github.com/tesseract-ocr/tesseract) and a python library [pytesseract](https://pypi.org/project/pytesseract/) was used. The result of OCR is automatically analysed and, if necessary, other image preprocessing parameters or other parameters of the OCR engine ware used