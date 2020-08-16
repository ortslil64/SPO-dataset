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
from spo_dataset.spo_generator import generate_dataset


    
        
        
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
    
