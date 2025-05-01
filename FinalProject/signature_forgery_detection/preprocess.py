# preprocess.py - Image Preprocessing for Signature Forgery Detection
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)
