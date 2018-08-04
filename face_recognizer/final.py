from faceID import faceID
from emotion_detection import create_model, emotionID
from camera_to_image_array import camera_to_image_array as cam
from dataface_funcs import dataface_add
from getting_name_dataface import match_descriptor_in_dataface as lookup
from image_array_to_descriptors import load_dlib as load
from image_array_to_descriptors import match_image_to_descriptors as match
from dataface_funcs import pickOpen

def faceRec():
    pic = cam()
    dataface = pickOpen()
    face_detect, face_rec_model, shape_predictor = load()
    out, rect = match(face_detect, face_rec_model, shape_predictor, pic)
    names = lookup(dataface, out)
    ax = faceID(pic, rect, names)
    emotionID(pic, ax, face_detect, shape_predictor, create_model())