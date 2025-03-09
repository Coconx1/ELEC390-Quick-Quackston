from aiymakerkit import vision
from pycoral.adapters.detect import BBox
from datetime import datetime
from moviepy import editor
import cv2
import numpy as np 

import moveipy

import numpy as np 
import pandas as pd 
import models

import time
import os.path



#DELAY_SECS = 3
#snap_time = 0
#PICTURE_DIR = os.path.join(os.path.expanduser('~'), 'Pictures')


prevLeft = prevRight = None



def process_feed( vision, vision_out):

    input_video = editor.VideoFileClip(test_video, audio=False)
   
    processed = input_video.fl_image(frame_processor)

    processed.write_videofile(output_video, audio=False)

def preprocessFrames(image):

    # make the pictures grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #gaussian kernel size
    kernel_size = 5

    #blur the frames to reduce the noise and smooth out the image
    blur =cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    #low threshold
    low_t= 50 

    #high threshold
    high_t = 150

    #detects edges using canny edge detection algorithm
    edges = cv2.Canny(blur, low_t, high_t)

    return edges


def maskRegion(image):

    #get the height of the mage
    height = image.shape[0]

    #draw a polygon over the ROI and mask it
    polygons = np.array([[(200, height), (1100,height), (550,250)]])

    mask = np.zeros_like(image)

    cv2.fillpoly(mask, polygons, 255)

    maked_image=cv2.bitwise_and(image, mask)

    #return the masked image
    return maskedImage


def getCoordinates(image, parameters):

    slope,intercept = parameters

    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int ((y1-intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array( [x1,y1,x2,y2] )


def findSlopeAverage (image, lines):

    global prevLeft, prevRight
    left_fit = []
    right_fit = []

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)

        parameter=np.polyfit((x1,x2), (y1,y2), 1)

        slope = parameter[0]

        intercept = parameter[1]

        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))




# def box_is_in_box(bbox_a, bbox_b):
#     if ((bbox_a.xmin > bbox_b.xmin) and (bbox_a.xmax < bbox_b.xmax)) and ((bbox_a.ymin > bbox_b.ymin) and (bbox_a.ymax < bbox_b.ymax)):
#         return True
#     return False

# for frame in vision.get_frames():
#     faces = detector.get_objects(frame, threshold=0.1)
    
#     face_in_box = 0
#     for face in faces:
#         if box_is_in_box(face.bbox, camera_bbox):
#             face_in_box += 1
    
#     if faces and face_in_box == len(faces) and time.monotonic() - snap_time > DELAY_SECS:
#         timestamp = datetime.now()
#         filename = "SMART_CAM_" + timestamp.strftime("%Y%m%d_%H%M%S") + '.png'
#         filename = os.path.join(PICTURE_DIR, filename)
#         vision.save_frame(filename, frame)
#         snap_time = time.monotonic()
#         print(filename)
#     else:
#         vision.draw_objects(frame, faces)
#         vision.draw_rect(frame, camera_bbox)