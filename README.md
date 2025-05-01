# Final Project README
# Document Forgery Detection System
Nishitha Reddy Kaluvala – U83727321
Laasya Chenchala – U12533113
## Project Overview
This project is a Document Forgery Detection System designed to identify two major types of forgeries:
1.	Signature Forgery Detection using an Autoencoder model (optionally supported by GAN-based synthetic data generation).
2.	Copy-Move Forgery Detection using a traditional block-based analysis approach.
Additionally, it includes a comprehensive Filter and Analysis module for preprocessing, text analysis, and signature isolation. The system incorporates trustworthy design principles — ensuring robustness, reliability, and transparency — by performing well across document distortions, using interpretable methods (reconstruction error, offset consistency), and modular architecture.
 
## Project Modules
### 1. Signature Forgery Detection
This module determines whether a given signature is genuine or forged by measuring how accurately an Autoencoder can reconstruct the input image. The Autoencoder is trained solely on genuine signature samples so that it learns the typical patterns and structures found in authentic signatures. During detection, a test signature is passed through the model. If the reconstruction error (the difference between the input and the output) is above a certain threshold, it is considered forged.
To improve training and expand the dataset, a Generative Adversarial Network (GAN) can be optionally used to generate synthetic genuine-looking signatures. This helps in cases where real signature samples are limited. The GAN learns from real signatures and generates fake ones that look realistic. These synthetic images can then be added to the training data for the Autoencoder.
Core files supporting this module include:
•	autoencoder_train.py for Autoencoder training,
•	gan_train.py for synthetic data generation,
•	detect_signature.py for signature verification,
•	preprocess.py for image conversion to grayscale, resizing, and normalization.
 
### 2. Copy-Move Forgery Detection
This module detects a specific type of forgery where a portion of a document is copied and pasted elsewhere within the same image. These forgeries are common in scanned documents and image-based manipulations. The algorithm begins by dividing the image into overlapping blocks. Each block is converted into a feature vector using dimensionality reduction methods such as Principal Component Analysis (PCA).
The extracted features from all blocks are then compared to identify blocks with high similarity. Matched blocks are further analyzed by calculating their relative spatial offsets. If multiple matching block pairs share consistent offsets, they likely belong to a copy-move forged region. The system then highlights these duplicated areas, flagging them as potential tampering zones.
Supporting files include:
•	Blocks.py for image division,
•	ImageObject.py and Container.py for block feature extraction and organization,
•	CopyMoveDetection.py for block comparison, offset analysis, and forgery localization,
•	main_GUI.py for the GUI interface that lets users upload images and view results.
 
### 3. Filter and Analysis Module
This module enhances the reliability and interpretability of the forgery detection system by isolating important document components such as signatures and text regions. It acts as a preprocessing and analysis bridge between raw scanned input and model evaluation.
In the OCR pipeline, the OCR.py script uses Tesseract OCR to extract readable text from scanned document images. This can be used to verify whether specific document fields (like names or dates) have been altered. The signature extraction process, implemented in signprocessing.py, uses thresholding to separate the signature ink from the background, then applies contour detection to isolate the signature area and enhance it for analysis.
The Text.py module detects and segments text regions within a document. It uses binary thresholding and morphological dilation to merge nearby characters into text blocks. Contour detection is then used to locate and crop these regions. This allows separate fields like "Name," "DOB," and "Signature" to be analyzed individually.
These preprocessing steps improve robustness by cleaning the input image, reducing background noise, and isolating specific features for targeted analysis, which increases the reliability and transparency of detection.
 
## How to Run the Project
### A. Signature Detection
Optional: Generate synthetic signatures
python gan_train.py

Train Autoencoder
python autoencoder_train.py

 Detect forgery on test images
python detect_signature.py

### B. Copy-Move Detection
Run GUI for copy-move analysis
python main_GUI.py
 
## Dependencies
•	Python 3.x
•	PyTorch
•	torchvision
•	OpenCV (cv2)
•	PIL (Pillow)
•	numpy
•	pytesseract (for OCR)
## Notes
•	Make sure all required folders (e.g., data/genuine/class1, models, generated) exist.
•	Ensure that pytesseract is correctly installed and configured for OCR.
•	For best GAN performance, train for at least 50+ epochs.





