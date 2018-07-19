import numpy as np

from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models

load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]

import skimage.io as io 

def image_file_to_image_array(path):
    """ Given an image file, load the file as an image array.
    
        Parameters
        ----------
        path : string
            describes the path of the image
                        
        Returns
        -------
        The image file as an image array. """
    
    return io.imread(path)

def load_image_to_descriptors(image, name):
    """ Given an image array with 1 known face, return the description
    vector of that face. 
    
        Parameters
        ----------
        image : numpy.ndarray, shape = (N, M)
            2D array of shape (N, M)
            
        name : string
            
        Returns
        -------
        A tuple containing the name and the descriptor of the face. """
    
    detections = list(face_detect(image))
    assert len(detections) == 1 
    
    shape = shape_predictor(image, detections[0])
    descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
    return (name, descriptor)

def match_image_to_descriptors(image):
    """ Given an image array with some amount of unknown faces, returns the 
    description vectors of each face present in the image.
        
        Parameters
        ----------
        image : numpy.ndarray, shape = (N, M)
            2D array of shape (N, M)
            
        Returns
        -------
        An array of the descriptors of all faces present in the image array. """
    
    detections = list(face_detect(image))
    out = []
    
    for face in range(len(detections)):
        shape = shape_predictor(image, detections[face])
        descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
        out.append(descriptor)
        
    return np.array(out)