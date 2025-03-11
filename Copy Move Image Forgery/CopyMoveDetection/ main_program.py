import CopyMoveDetection
import cv2
from PIL import Image

"""
Main Code - Copy-Move Forgery Detection
"""

# To detect copy-move forgery in all images under a directory, use detect_dir()
# This will process all images in the given folder and save results in the output directory
# Example usage:
# CopyMoveDetection.detect_dir('../testcase_image/', '../testcase_result/', 32)

# To detect copy-move forgery in a single image, use detect()
# This processes a specific image and saves the result
# Example usage:
# CopyMoveDetection.detect('../testcase_image/', '01_barrier_copy.png', '../testcase_result/', blockSize=32)

# Example of preprocessing an image before detection
from PIL import Image  # Import Image module from PIL

# Open the image in grayscale mode with transparency (LA = Luminance + Alpha)
img = Image.open('test_images/Test-2.png').convert('LA')

# Get the image dimensions
width, height = img.size  

# Resize the image to half its original size for processing optimization
img = img.resize((int(width / 2), int(height / 2)))

# Save the optimized image with reduced size and better quality
img.save('test_images/Test-2.png', optimize=True, quality=95)

# Perform Copy-Move Forgery Detection on the preprocessed image
# - 'test_images/' is the directory where the input image is stored
# - 'Test-2.png' is the target image for detection
# - 'results/' is the output directory where results will be saved
# - blockSize=32 defines the size of the blocks used in forgery detection
CopyMoveDetection.detect('test_images/', 'Test-2.png', 'results/', blockSize=32)