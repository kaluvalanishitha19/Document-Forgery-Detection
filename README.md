# Document-Forgery-Detection

Overview

This project aims to detect forged documents using advanced image processing and machine learning techniques. It helps in identifying tampered or manipulated documents by analyzing inconsistencies in text, signatures, stamps, or other visual elements.

Signature Forgery Detection:
Signature forgery detection involves identifying whether a signature on a document is genuine or forged.
Image Preprocessing – Noise removal, thresholding, cropping.
Feature Extraction – Converts signatures into feature vectors.
Neural Network Training – Deep learning model identifies genuine vs. forged.
Forgery Classification – Predicts forgery based on feature similarity.

Copy-Move Image Manipulation Detection:

The process of detecting tampered regions in an image where a part of the image is copied and pasted elsewhere within the same image, typically using keypoint detection and pattern matching methods.
Block-Based Image Segmentation – Divides image into overlapping blocks.
Feature Extraction Using PCA – Dimensionality reduction for efficiency.
Feature Matching – Finds duplicated regions in the image.
Forgery Localization & Visualization – Highlights copied regions. 


