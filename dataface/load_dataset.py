from image_file_to_image_array import image_file_to_image_array
from image_array_to_descriptors import load_image_to_descriptors
import numpy as np
import os

def load_dataset(dataset="data\ours", extension=".JPG"):
    '''
    Loads and returns all images in a dataset
    
    INPUTS:
    
        dataset:        Path to the top of the dataset
                        string path
        
        extension:      The extension of images in the database
                        All files not ending with the given extension are ignored
                        string
    
    OUPUTS:
    
        descriptors:    a collection of the descriptors of every successful image
                        ndarray shape (N,128)
        
        truth:          groundtruth labels of the descriptors
                        list of length N
        
        names:          names of people, taken from folder names
                        list of length N
    '''
    descriptors = []
    truth = []
    names = []
    label = 0
    # find all class folders
    for di in os.listdir(dataset):
        dire = os.path.join(dataset, di)
        if os.path.isdir(dire):
            # find individual images
            for fi in os.listdir(dire):
                if fi.endswith(extension):
                    file = os.path.join(dire, fi)
                    img = image_file_to_image_array(file)
                    try:
                        name, descriptor = load_image_to_descriptors(img, di)
                        descriptors.append(descriptor)
                        truth.append(label)
                        names.append(name)
                    except AssertionError:
                        print("Bad image: " + file)
            label += 1
    return np.array(descriptors), truth, names
