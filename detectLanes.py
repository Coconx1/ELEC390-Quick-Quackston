from aiymakerkit import vision
from pycoral.adapters.detect import BBox
from datetime import datetime
from moviepy import editor
import cv2

import moveipy

import numpy as np 
import pandas as pd 
import models

import time
import os.path

detector = vision.Detector(models.FACE_DETECTION_MODEL) # need to make the road line model for line detection

DELAY_SECS = 3
snap_time = 0
PICTURE_DIR = os.path.join(os.path.expanduser('~'), 'Pictures')




def process_feed( vision, vision_out):

    input_video = editor.VideoFileClip(test_video, audio=False)
    # apply the function "frame_processor" to each frame of the video
    # will give more detail about "frame_processor" in further steps
    # "processed" stores the output video
    processed = input_video.fl_image(frame_processor)

    # save the output video stream to an mp4 file
    processed.write_videofile(output_video, audio=False)

def preprocessFrames(image):

    # make the pictures grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #gaussian kernel size
    kernel_size = 5

    #blur the frames to reduce the noise
    blur =cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    low_t= 50 

    high_t = 150

    edges = cv2.Canny(blur, low_t, high_t)

    region = region_selection(edges)




    






def box_is_in_box(bbox_a, bbox_b):
    if ((bbox_a.xmin > bbox_b.xmin) and (bbox_a.xmax < bbox_b.xmax)) and ((bbox_a.ymin > bbox_b.ymin) and (bbox_a.ymax < bbox_b.ymax)):
        return True
    return False

for frame in vision.get_frames():
    faces = detector.get_objects(frame, threshold=0.1)
    
    face_in_box = 0
    for face in faces:
        if box_is_in_box(face.bbox, camera_bbox):
            face_in_box += 1
    
    if faces and face_in_box == len(faces) and time.monotonic() - snap_time > DELAY_SECS:
        timestamp = datetime.now()
        filename = "SMART_CAM_" + timestamp.strftime("%Y%m%d_%H%M%S") + '.png'
        filename = os.path.join(PICTURE_DIR, filename)
        vision.save_frame(filename, frame)
        snap_time = time.monotonic()
        print(filename)
    else:
        vision.draw_objects(frame, faces)
        vision.draw_rect(frame, camera_bbox)