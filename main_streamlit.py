import streamlit as st
import torch
#from detect import *
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
#import wget
import time
import pytesseract
import shutil
import cv2
import numpy as np

cfg_model_path = "weights/car_plates.pt" 

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

# def img_preprocessor01(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
#     mask = np.zeros(img.shape, dtype=np.uint8)
#     cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cv2.fillPoly(mask, cnts, [255,255,255])
#     mask = 255 - mask
#     result = cv2.bitwise_or(img, mask)
#     return result

def img_preprocessor01(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.medianBlur(img,5)
    #img = cv2.GaussianBlur(img,(5,5),0)
    #_, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #_, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    #Remove image borders
    mask = np.zeros(img.shape, dtype=np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, [255,255,255])
    mask = 255 - mask
    binaryImage2 = cv2.bitwise_or(img, mask)
    binaryImage2 = cv2.erode(binaryImage2, (3,3),iterations = 2)
    #binaryImage2 = cv2.dilate(binaryImage2, (4,4),iterations = 3)
    img_byArea = cv2.bitwise_not(areaFilter(80, cv2.bitwise_not(binaryImage2)))
    result = only_high_contours(img_byArea, 0.45)
    return result

def txt_postproc01(text):
    ln = text
    lines = text.split('\n')
    if len(lines)>1:
        ln = max(lines, key=len)
        if len(ln)<4:
            ln='-1'
    if (len(lines)==1 and len(lines[0])<4) or (len(lines)==0):
        ln = '-1'
    return ln


def imageInput(device, src):
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            
            colA,colB = st.columns(2)
            with colB:
                ta = st.empty()
                tb = st.empty()           
            col1, col2 = st.columns(2)
        
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            #model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
            model = torch.hub.load('Dimarfin/License_plate_recognition',
                                   #'D:\Dima\DataScience\Projects\Cars_plate_recognition\Code\Yolov5\yolov5', 
                                   'custom', 
                                   #source='local', 
                                   path=cfg_model_path, 
                                   force_reload=True, 
                                   device='cpu') 
            #model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.imgs:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')
                crops = pred.crop(save=True)
                crop_path = 'runs/detect/exp/crops/License/'
                crops = os.listdir(crop_path)
                
                for crop in crops:
                    img = cv2.imread(crop_path+crop)
                    text = pytesseract.image_to_string(img_preprocessor01(img), config = '--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    text = txt_postproc01(text)
                    if text=='-1':
                        text = pytesseract.image_to_string(img, config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if text=='-1':
                        text = pytesseract.image_to_string(img, config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    ta.write("Licence plate(s) text: ")
                    tb.subheader(text)
                shutil.rmtree('runs/detect/exp/')
                
    elif src == 'From test set.': 
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select an image from the test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Detect!")
        #st.write(image_file)
        
        colA,colB = st.columns(2)
        with colB:
            ta = st.empty()
            tb = st.empty()           
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                #model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
                model = torch.hub.load('Dimarfin/License_plate_recognition',
                                      #'D:\Dima\DataScience\Projects\Cars_plate_recognition\Code\Yolov5\yolov5', 
                                      'custom', 
                                      #source='local', 
                                      path=cfg_model_path, 
                                      force_reload=True, 
                                      device='cpu') 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.imgs:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')
                crops = pred.crop(save=True)
                crop_path = 'runs/detect/exp/crops/License/'
                crops = os.listdir(crop_path)
                #st.write("Licence plate(s) text:")
                for crop in crops:
                    img = cv2.imread(crop_path+crop)
                    text = pytesseract.image_to_string(img_preprocessor01(img), config = '--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    text = txt_postproc01(text)
                    if text=='-1':
                        text = pytesseract.image_to_string(img, config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    if text=='-1':
                        text = pytesseract.image_to_string(img, config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    ta.write("Licence plate(s) text: ")
                    tb.subheader(text)
                shutil.rmtree('runs/detect/exp/')



def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
        outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        #detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(weights=cfg_model_path, source=imgpath, device='cpu')
        st_video2 = open(outputpath, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.write("Model Prediction")


def main():
    # -- Sidebar
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\kater\AppData\Local\Tesseract-OCR\tesseract.exe'
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['From test set.', 'Upload your own data.'])
    
        
    option = "Image" 
    deviceoption = 'cpu'
    #option = st.sidebar.radio("Select input type.", ['Image', 'Video'], disabled = True, index=0)
    #if torch.cuda.is_available():
    #    deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    #else:
    #    deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar
    
    
    st.header('🚘 Car license plate recognition system')
    st.subheader('👈🏽 Select options from the left-hand menu bar')
    #st.sidebar.markdown("https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")
    if option == "Image":    
        imageInput(deviceoption, datasrc)
        # image_file = "data/images\Cars34.png"
        # img = Image.open(image_file)
        # st.image(img, caption='Selected Image', use_column_width='always')
        # model = torch.hub.load('D:\Dima\DataScience\Projects\Cars_plate_recognition\Code\Yolov5\yolov5', 
        #                        'custom', 
        #                        source='local', 
        #                        path=cfg_model_path, 
        #                        force_reload=True, 
        #                        device='cpu') 
        # pred = model(image_file)
        # pred.render()
        # for im in pred.imgs:
        #     im_base64 = Image.fromarray(im)
        #     im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
        #    #st.image(im, caption='Model Prediction(s)')
    elif option == "Video": 
        videoInput(deviceoption, datasrc)


if __name__ == '__main__':
  
    main()

