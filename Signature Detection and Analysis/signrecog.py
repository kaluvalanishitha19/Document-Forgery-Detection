import cv2
import os
import numpy as np
import network
import preprocessor


def main():
    """ Main function to train and evaluate a neural network for signature verification. """

    # Print OpenCV version to confirm installation
    print('OpenCV version {} '.format(cv2.__version__))

    # Get the current directory of the script
    script_directory = os.path.dirname(__file__)

    # Define paths for training and test datasets
    training_folder = os.path.join(script_directory, 'data/training/')
    test_folder = os.path.join(script_directory, 'data/test/')

    # Initialize an empty list to store training data
    training_data = []
    
    # Iterate through images in the training folder
    for file_name in os.listdir(training_folder):
        image_path = os.path.join(training_folder, file_name)
        
        # Read the image in grayscale mode
        img = cv2.imread(image_path, 0)
        
        if img is not None:
            # Extract and preprocess image features
            extracted_features = np.array(preprocessor.prepare(img))
            reshaped_features = np.reshape(extracted_features, (901, 1))  # Reshape for neural network input
            
            # Assign labels: [0,1] for genuine signatures, [1,0] for forged signatures
            label = [[0], [1]] if "genuine" in file_name else [[1], [0]]
            label = np.array(label)
            label = np.reshape(label, (2, 1))
            
            # Append the processed image and label to the training dataset
            training_data.append((reshaped_features, label))

    # Initialize an empty list to store test data
    test_data = []
    
    # Iterate through images in the test folder
    for file_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, file_name)
        
        # Read the image in grayscale mode
        img = cv2.imread(image_path, 0)
        
        if img is not None:
            # Extract and preprocess image features
            extracted_features = np.array(preprocessor.prepare(img))
            reshaped_features = np.reshape(extracted_features, (901, 1))  # Reshape for neural network input
            
            # Assign labels: 1 for genuine, 0 for forged
            label = 1 if "genuine" in file_name else 0
            
            # Append the processed image and label to the test dataset
            test_data.append((reshaped_features, label))

    # Initialize a neural network with 901 input neurons, two hidden layers (500 neurons each), and 2 output neurons
    net = network.NeuralNetwork([901, 500, 500, 2])

    # Train the neural network using stochastic gradient descent (SGD)
    # - 10 epochs
    # - Batch size of 50
    # - Learning rate of 0.01
    net.sgd(training_data, 10, 50, 0.01, test_data)

    # Evaluate the trained model using test data and print the accuracy
    print(net.evaluate(test_data))


# Ensure the script runs only when executed directly
if __name__ == '__main__':
    main()