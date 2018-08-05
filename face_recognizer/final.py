from face_recognizer.faceID import faceID
from face_recognizer.emotion_detection import create_model, emotionID
from face_recognizer.camera_to_image_array import camera_to_image_array as cam
from face_recognizer.dataface_funcs import dataface_add
from face_recognizer.getting_name_dataface import match_descriptor_in_dataface as lookup
from face_recognizer.image_array_to_descriptors import load_dlib as load
from face_recognizer.image_array_to_descriptors import match_image_to_descriptors as match
from face_recognizer.dataface_funcs import pickOpen

def faceRec():
    pic = cam()
    dataface = pickOpen(path = 'dataface/dataface.dat')
    face_detect, face_rec_model, shape_predictor = load()
    out, rect = match(face_detect, face_rec_model, shape_predictor, pic)
    names = lookup(dataface, out)
    ax = faceID(pic, rect, names)
    emotion = emotionID(pic, ax, face_detect, shape_predictor, create_model())
    return emotion

def faceRec2(img_arr):
    """ Performs face and emotion recognition on an image 
    array, rather than on a picture taken from the camera.
        
        Parameters
        ----------
        img_arr : nd.array
            the image, as an array
        
        Returns
        -------
        The emotion. """    
    
    dataface = pickOpen(path = 'dataface/dataface.dat')
    face_detect, face_rec_model, shape_predictor = load()
    out, rect = match(face_detect, face_rec_model, shape_predictor, img_arr)
    names = lookup(dataface, out)
    ax = faceID(img_arr, rect, names)
    emotion = emotionID(img_arr, ax, face_detect, shape_predictor, create_model())
    return emotion
