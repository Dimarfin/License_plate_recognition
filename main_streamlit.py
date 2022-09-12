#Acknowledgments 
#- https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment
#- https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python
import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import time
import pytesseract
import shutil
import cv2
import numpy as np

cfg_model_path = "weights/car_plates.pt" 

def remove_borders(img):
    _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, [255,255,255])
    mask = 255 - mask
    result = cv2.bitwise_or(img, mask)
    return result

def img_preprocessor01(img):
    img = img[:,:,0]#Remove blue channel
    #
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img = remove_borders(img)
    #result = cv2.erode(result, (3,3),iterations = 2)
    img = cv2.dilate(img, (4,4),iterations = 3)
    result = cv2.bitwise_not(areaFilter(30, cv2.bitwise_not(img)))
    result = only_high_contours(result, 0.35/(3.5*img.shape[0]/img.shape[1]))
    return result

def img_preprocessor03(img):
    _, img = cv2.threshold(img[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = remove_borders(img)
    result = cv2.bitwise_not(areaFilter(30, cv2.bitwise_not(img)))
    result = only_high_contours(result, 0.35/(3.5*img.shape[0]/img.shape[1]))
    #result = cv2.erode(result, (3,3),iterations = 2)
    #result = cv2.dilate(result, (4,4),iterations = 3)
    return result

def img_preprocessor04(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img = remove_borders(img)
    result = cv2.bitwise_not(areaFilter(80, cv2.bitwise_not(img)))
    result = only_high_contours(result, 0.45/(3.5*img.shape[0]/img.shape[1]))
    #result = cv2.erode(result, (3,3),iterations = 2)
    #result = cv2.dilate(result, (4,4),iterations = 3)
    return result

def areaFilter(minArea, inputImage):
    #Function from https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def only_high_contours(img, hight):
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    cnts1 = []
    for c in cnts:
      top_point = tuple(c[c[:,:,1].argmin()][0])
      bottom_point = tuple(c[c[:,:,1].argmax()][0])
      h = bottom_point[1] - top_point[1]
                
      if h > hight*img.shape[0]:
          cnts1 = cnts1 + [c]
    
    cv2.fillPoly(mask, cnts1, [255,255,255])
    result = cv2.bitwise_or(img, mask)
    return result

def img_preprocessor_main(img, a=30, h=0.35, mode='rm_blue',threshold='adaptive'):
    if mode=='rm_blue':
        img = img[:,:,0]
    elif mode=='to_gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if threshold=='adaptive':    
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    elif threshold=='otsu':
        _, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    img = remove_borders(img)
    result = cv2.bitwise_not(areaFilter(a, cv2.bitwise_not(img)))
    result = only_high_contours(result, h/(3.5*img.shape[0]/img.shape[1]))
    result = cv2.erode(result, (3,3),iterations = 1)
    #result = cv2.dilate(result, (3,3),iterations = 1)
    return result

def do_OCR(crop_path):
    ts_cfg = '--psm 1 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    crops = os.listdir(crop_path)
    text = []
    for crop in crops:
        img = cv2.imread(crop_path+crop)
        a=30
        h=0.35
        result = img_preprocessor_main(img, a, h, mode='rm_blue',threshold='otsu')
        tx = pytesseract.image_to_string(result, config = ts_cfg)
        if len(tx.strip())<4:
            result = img_preprocessor_main(img, a, h, mode='rm_blue',threshold='adaptive')
            tx = pytesseract.image_to_string(result, config = ts_cfg)
        if len(tx.strip())<4:
            result = img_preprocessor_main(img, a, h, mode='to_gray',threshold='adaptive')
            tx = pytesseract.image_to_string(result, config = ts_cfg)
        if len(tx.strip())<4:
            result = img_preprocessor_main(img, a, h, mode='to_gray',threshold='otsu')
            tx = pytesseract.image_to_string(result, config = ts_cfg)
        if len(tx.strip())<4:
            tx = pytesseract.image_to_string(img, config = ts_cfg)
        text = text + [tx]
    shutil.rmtree('runs/detect/exp/')
    return text

def call_model_prediction(imgpath,outputpath):
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
    model = torch.hub.load('Dimarfin/License_plate_recognition',
                             'custom',  
                             path=cfg_model_path, 
                             force_reload=True, 
                             device='cpu')
    pred = model(imgpath)
    pred.render()  # render bbox in image
    crops = pred.crop(save=True)
    for im in pred.imgs:
        im_base64 = Image.fromarray(im)
        im_base64.save(outputpath)

def detect_and_show(submit,image_file,imgpath,outputpath):
    if image_file is not None:
        img = Image.open(image_file)
    
        colA,colB = st.columns(2)
        with colB:
            ta = st.empty()
            tb = st.empty()           
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='Input Image', use_column_width='always')
               
        if submit:
            #call Model prediction--
            call_model_prediction(imgpath,outputpath)
            #--Display predicton
            img_ = Image.open(outputpath)
            with col2: 
                st.image(img_, caption='Model Prediction(s)')
                crop_path = 'runs/detect/exp/crops/License/'
                text = do_OCR(crop_path)
                ta.write("Licence plate(s) text: ")
                text_line = ''
                for tx in text:
                    text_line = text_line + ' ' + tx
                tb.subheader(text_line.strip())    

def imageInput(device, src):
    
    if src == 'Upload your own image.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        submit = st.button("Detect!")
        if image_file is not None:
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            
            detect_and_show(submit,image_file,imgpath,outputpath)
                
    elif src == 'Choose an image from test set.': 
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select an image from the test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        outputpath = os.path.join('data/outputs', os.path.basename(image_file))
        submit = st.button("Detect!")
        
        detect_and_show(submit,image_file,image_file,outputpath)
                
    elif src == 'Show project description.':
        #st.header('Project description')
        st.write('This model is the final project of the Data Science Boot Camp at the WBS coding school, Berlin. The source code can be found on GitHub [https://github.com/Dimarfin/License_plate_recognition](https://github.com/Dimarfin/License_plate_recognition)')
        st.markdown('''
                    ### Model implementation
                    This model implements detection the car license plate and recognition of 
                    the text on it. It includes several steps depicted in the figure below. 
                    The first two (training and validation) were performed only once while
                    the following steps are repeated each time a new image is supplied to the model.
                    ''')
        fig01 = Image.open('docs/fig01.png')
        st.image(fig01, width = 400, caption='Schematic representation of the model pipeline')
        st.markdown('''
                    ### Object detection step
                    The detection step is performed using [YOLOv5](https://github.com/ultralytics/yolov5) – a state of the art object detection system. It is based on a constitutional neuronal network and build using PyTorch library. YOLOv5 is supplied pertained on MS COCO dataset which speeds up the training step for a custom object type. One of the review of the YOLO system can be found at the following [link](https://towardsdatascience.com/yolo-v5-is-here-b668ce2a4908). 
                    While there are several sizes of the YOLOv5 in this work YOLOv5s was utilized. For detection of the car license plates it was trained on a [data set from Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).  
                    The model reached outstanding performance after 20 epochs of training. Some of the performance metrics measured on the validation subset are presented in the figure below.
                    ''')
        fig02 = Image.open('docs/fig02.png')             
        st.image(fig02, caption='Model performance') 
        st.markdown(''' 
                     ### Image processing step
                     After detection of the license plate the image is cropped around it. 
                     Before sending this croped image to OCR the image preprocessing step 
                     is applied. The image preprocessing is realized using [openCV library](https://opencv.org/) and
                     includes several operation such as removing certain colors, thresholding, 
                     filtering, contour detection and clearing unsuitable ones. There are several 
                     oreprocessors realized withing this project. If OCR result achieved after
                     one preprocessor doesn't match certain criteria, another preprocessor is used
                     and the OCR step is repeated'
                     
                     ### Optical character recognition
                     To recognized text on the license plate [Tesseract Open Source OCR Engine](https://github.com/tesseract-ocr/tesseract) 
                     and a python library [pytesseract](https://pypi.org/project/pytesseract/) was used. 
                     The result of OCR is automatically analysed and, if necessary, another image 
                     preprocessor is used. At the current stage of developtment the character 
                     error rate (CER) measured on a [test data set](https://github.com/Dimarfin/License_plate_recognition/tree/main/ocr_tune)
                     appeared to be equal 11.4%.
                     ''')

def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("", 
                               ['Choose an image from test set.', 
                                'Upload your own image.',
                                'Show project description.'])
         
    st.sidebar.write('[GitHub link](https://github.com/Dimarfin/License_plate_recognition)')
    # -- End of Sidebar
    
    
    st.header('🚘 Car license plate recognition system')
    st.subheader('👈🏽 Select options from the left-hand menu bar')
    st.markdown("""---""")
    imageInput('cpu', datasrc)
    

if __name__ == '__main__':
  
    main()
