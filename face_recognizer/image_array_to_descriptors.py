import numpy as np

def load_dlib():
    """ Loads dlib models. 
    
        Returns
        -------
        Three dlib models: face detect, face rec, and shape predict. """

    from dlib_models import download_model, download_predictor, load_dlib_models
    download_model()
    download_predictor()
    from dlib_models import models

    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]
    return (face_detect, face_rec_model, shape_predictor)

def load_image_to_descriptors(face_detect, face_rec_model, shape_predictor, image, name):
    """ Given an image array with 1 known face, return the description
    vector of that face. 
    
        Parameters
        ----------
        face_detect : dlib model
        
        face_rec_model : dlib model
        
        shape_predictor : dlib model
        
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

def match_image_to_descriptors(face_detect, face_rec_model, shape_predictor, image):
    """ Given an image array with some amount of unknown faces, returns the 
    description vectors of each face present in the image.
        
        Parameters
        ----------
        face_detect : dlib model
        
        face_rec_model : dlib model
        
        shape_predictor : dlib model
        
        image : numpy.ndarray, shape = (N, M)
            2D array of shape (N, M)
            
        Returns
        -------
        An array of the descriptors of all faces present in the image array
        and the list of dlib rectangles for detected faces. """
    
    detections = list(face_detect(image))
    out = []
    
    for face in range(len(detections)):
        shape = shape_predictor(image, detections[face])
        descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))
        out.append(descriptor)
        
    return (np.array(out), detections)
