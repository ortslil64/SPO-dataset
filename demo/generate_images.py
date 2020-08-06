#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage import data
from skimage.transform import warp
import cv2
import time
from tqdm import tqdm
import skvideo.io
import imutils
import dlib
from imutils import face_utils
from spo_dataset.spo_generator import get_video, get_dataset_from_video, get_dataset_from_image, generate_image

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
        
    
        
        
if __name__ == '__main__':
    generate_dataset(n = 1000,
                      video_path='../spo_dataset/source_video/water.mp4',
                      visualize=False,
                      output_folder="dataset/images/water/",
                      output_type="images")
    
    generate_dataset(n = 1000,
                      video_path='../spo_dataset/source_video/water.mp4',
                      visualize=False,
                      output_folder="dataset/video/water/",
                      output_type="video")
    
    generate_dataset(n = 100,
                      video_path='../spo_dataset/source_video/illusion.mp4',
                      visualize=False,
                      output_folder="dataset/images/illusion/",
                      output_type="images")
    
    generate_dataset(n = 100,
                      image_type = "dots",
                      visualize=False,
                      output_folder="dataset/video/dots/",
                      output_type="video")
    
    generate_dataset(n = 100,
                      image_type = "dots",
                      visualize=False,
                      output_folder="dataset/images/dots/",
                      output_type="images")
    
    generate_dataset(n = 100,
                      image_type = "checkers",
                      visualize=False,
                      output_folder="dataset/video/checkers/",
                      output_type="video")
    
    generate_dataset(n = 100,
                      image_type = "checkers",
                      visualize=False,
                      output_folder="dataset/images/checkers/",
                      output_type="images")
    
    generate_dataset(n = 100,
                      image_type = "checkers",
                      visualize=False,
                      output_folder="dataset/video/checkers_partial/",
                      output_type="video",
                      partial=True)
    
    generate_dataset(n = 100,
                      image_type = "checkers",
                      visualize=False,
                      output_folder="dataset/images/checkers_partial/",
                      output_type="images",
                      partial=True)
    
    generate_dataset(n = 100,
                     image_path='../spo_dataset/source_image/tree.jpg',
                     mask='../spo_dataset/source_image/tree_masked.jpg',
                     visualize=False,
                     output_folder="dataset/video/tree_partial/",
                     output_type="video",
                     partial=True)
    
    generate_dataset(n = 100,
                     image_path='../spo_dataset/source_image/tree.jpg',
                     mask='../spo_dataset/source_image/tree_masked.jpg',
                     visualize=False,
                     output_folder="dataset/images/tree_partial/",
                     output_type="images",
                     partial=True)
    
