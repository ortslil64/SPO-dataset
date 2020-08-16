#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.transform import warp
import cv2
import time
from tqdm import tqdm
import skvideo.io
import imutils
import dlib
from imutils import face_utils

def drc(xy,c_xy,radius):
    xy_output = xy
    for ii in range(len(xy)):
        r = np.sqrt((xy[ii,0]-c_xy[0])**2 + (xy[ii,1]-c_xy[1])**2)
        if r < radius:
            v = c_xy - xy[ii]
            if np.linalg.norm(v) > 0:
                v = v/np.linalg.norm(v)
                xy_output[ii,:] = xy[ii,:] + radius*v*(np.exp(-0.0001*r))
    return xy_output

def add_circle_mag(image, c_xy, radius):
    warp_func = lambda xy: drc(xy,c_xy,radius)
    return warp(image, warp_func)

def get_dataset_from_image(image, n = 500, n_circles = 2, radius = None, v = None, pose = None, partial = False, mask = None):
    x = []
    z = []
    
   
    possible_speeds = [-3, -2, -1, 1, 2, 3]
    if radius is None:
        radius = []
        for ii in range(n_circles):
            radius_temp = np.random.randint(10,30, dtype=np.int8)
            radius.append(radius_temp)
    
    if v is None:
        v = []
        for ii in range(n_circles):
            v_temp = np.random.choice(possible_speeds, size = 2)
            v.append(v_temp)
         
    if pose is None: 
        pose = []
        for ii in range(n_circles):
            pose_temp = np.empty(2)
            pose_temp[0] = np.random.randint(0, image.shape[0], dtype=np.int8)
            pose_temp[1] = np.random.randint(0, image.shape[1], dtype=np.int8)
            pose.append(pose_temp)
        
        
    
    for ii in tqdm(range(n)):
        swirled = image.copy()
        state = np.zeros_like(swirled)
        for jj in range(n_circles):
            pose[jj][0] = np.int32(pose[jj][0] + v[jj][0])
            pose[jj][1] = np.int32(pose[jj][1] + v[jj][1])
            if pose[jj][0] > image.shape[0] -1:
                pose[jj][0] = image.shape[0] -1
                v[jj][0] = -v[jj][0]
            if pose[jj][1] > image.shape[1]-1:
                pose[jj][1] = image.shape[1]-1
                v[jj][1] = -v[jj][1]
            if pose[jj][0] < 1:
                pose[jj][0] = 1
                v[jj][0] = -v[jj][0]
            if pose[jj][1] < 1:
                pose[jj][1] = 1
                v[jj][1] = -v[jj][1]
            
            if partial == True and ii > 0 and mask is None:
                swirled  = add_circle_mag(swirled,  pose[jj], radius[jj])
                swirled[20:90,20:90] = 0.5
                # if pose[jj][0] < 30 or pose[jj][0] > 80 or pose[jj][1] < 30 or pose[jj][1] > 80:
                #     swirled  = add_circle_mag(swirled,  pose[jj], radius[jj])
            elif mask is not None:
                if mask[int(pose[jj][1]),int(pose[jj][0])]<255:
                    swirled  = add_circle_mag(swirled,  pose[jj], radius[jj])
            else:
                swirled  = add_circle_mag(swirled,  pose[jj], radius[jj])
            state = cv2.circle(state,(int(pose[jj][0]), int(pose[jj][1])),radius[jj],(255,255,255),-1) 
        x.append(state//255.0)
        z.append(swirled)
    x = np.array(x)
    z = np.array(z)
    return x,z


def get_dataset_from_video(images, n = 500, n_circles = 2, radius = None):
    x = []
    z = []
    v = []
    pose = []
    if radius is None:
        radius = []
        for ii in range(n_circles):
            radius_temp = np.random.randint(10,30, dtype=np.int8)
            radius.append(radius_temp)  
    image = images[0]
    possible_speeds = [-3, -2, -1, 1, 2, 3]
    if len(images) < n:
	    n = len(images)
	
    for ii in range(n_circles):
        pose_temp = np.empty(2)
        pose_temp[0] = np.random.randint(0, image.shape[0], dtype=np.int8)
        pose_temp[1] = np.random.randint(0, image.shape[1], dtype=np.int8)
        pose.append(pose_temp)
        v_temp = np.random.choice(possible_speeds, size = 2)
        v.append(v_temp)

    
    for ii in tqdm(range(n)):
        image = images[ii]
        swirled = image.copy()
        state = np.zeros_like(swirled)
        for jj in range(n_circles):
            pose[jj] = np.int32(pose[jj] + v[jj])
            if pose[jj][0] > image.shape[0] -1:
                pose[jj][0] = image.shape[0] -1
                v[jj][0] = -v[jj][0]
            if pose[jj][1] > image.shape[1]-1:
                pose[jj][1] = image.shape[1]-1
                v[jj][1] = -v[jj][1]
            if pose[jj][0] < 1:
                pose[jj][0] = 1
                v[jj][0] = -v[jj][0]
            if pose[jj][1] < 1:
                pose[jj][1] = 1
                v[jj][1] = -v[jj][1]
            swirled  = add_circle_mag(swirled,  pose[jj], radius[jj])
            state = cv2.circle(state,(pose[jj][0], pose[jj][1]),radius[jj],(255,255,255),-1) 
        x.append(state//255.0)
        z.append(swirled)
    x = np.array(x)
    z = np.array(z)
    return x,z

def generate_image(r = 0.1):
    image = np.zeros((128,128))
    for ii in range(128):
        for jj in range(128):
            image[ii,jj] = np.random.binomial(1,r)
    return image

def generate_deterministic_image(n_x = 10, n_y = 10):
    image = np.zeros((128,128))
    for ii in range(128):
        for jj in range(128):
            if ii % n_x == 0 and jj % n_y == 0: 
                image[ii,jj] = 1
    return image
    
def get_video(video_path, n_frames = None):
    cap = cv2.VideoCapture(video_path)
    images_gray = []
    images_color = []
    n_images = 0
    if n_frames is None:
        while(cap.isOpened()):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray/255.0
            gray = cv2.resize(gray,(128,128))
            col = cv2.resize(frame,(128,128))
            col = col/255.0
            images_color.append(col)
            images_gray.append(gray)
    else:
        while(cap.isOpened() and n_images < n_frames):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray/255.0
            gray = cv2.resize(gray,(128,128))
            col = cv2.resize(frame,(128,128))
            col = col/255.0
            images_color.append(col)
            images_gray.append(gray)
            n_images += 1
    cap.release()
    return images_gray, images_color


def video2dataset(observation_video_path, frame_size = (256,256)):
    observation_images = []
    observation_cap = cv2.VideoCapture(observation_video_path)
    while(observation_cap.isOpened()):
        ret, frame = observation_cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray/255.0
        gray = cv2.resize(gray,frame_size)
        observation_images.append(gray)
    observation_cap.release()

   
    return  np.array(observation_images)

def face_detection(video_path, frame_size = (256,256)):
    name_list = ['mouth', 'left_eyebrow', 'right_eyebrow']
    state_cap = cv2.VideoCapture(video_path)
    state_cap = cv2.VideoCapture(video_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('source_video/facial_expression/shape_predictor_68_face_landmarks.dat')
    state_images = []
    while(state_cap.isOpened()):
        ret, frame = state_cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        blank = np.zeros_like(gray)
        for (i, rect) in enumerate(rects):
        	# determine the facial landmarks for the face region, then
        	# convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            	if name in name_list:
                    for (x, y) in shape[i:j]:
                       cv2.circle(blank, (x, y), 1, 255, -1)

        blank = blank/255.0
        blank = cv2.resize(blank,frame_size)
        state_images.append(blank)
        
    return np.array(state_images)


def generate_dataset(n = 100,video_path = None, image_path = None, image_type = None, output_path = None, visualize = False, output_type = "images", output_folder = "dataset/images/dots/",  partial = False, mask = None):
    frames = []
    if mask is not None:
        mask = cv2.imread(mask,0)
        mask = cv2.resize(mask, (128,128),interpolation = cv2.INTER_AREA)
    if video_path is not None:
        images, _ = get_video(video_path, n)
        x,z = get_dataset_from_video(images, n)
    elif image_path is not None:
        image = cv2.imread(image_path,0)
        image = cv2.resize(image, (128,128),interpolation = cv2.INTER_AREA)
        x,z = get_dataset_from_image(image/255, n, radius = [15, 25],  partial = partial, mask = mask)
    elif image_type == "dots":
        image = generate_image(0.01)
        x,z = get_dataset_from_image(image, n, radius = [15, 25],  partial = partial, mask = mask)
    elif image_type == "checkers":
        image = np.array(data.checkerboard()).astype(np.float64)
        image = cv2.resize(image, (128,128),interpolation = cv2.INTER_AREA)
        x,z = get_dataset_from_image(image/255, n, radius = [15, 25],  partial = partial, mask = mask)
    
    for ii in range(n):
        swirled  = z[ii]
        state = x[ii]
        if visualize:
            cv2.imshow('swirled',swirled)
            cv2.imshow('state',state)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            time.sleep(0.01)
        frame = np.concatenate((state,swirled),axis = 1)
        if output_type == "video":
            frames.append(frame*255)
        elif output_type == "images":
            fname = output_folder+str(ii)+".jpg"
            skimage.io.imsave(fname,frame*255)
    if visualize:
        cv2.destroyAllWindows()
        
    if output_type == "video":
        fname = output_folder+"dataset.mp4"
        skvideo.io.vwrite(fname, frames) 
        

    
