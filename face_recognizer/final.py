from faceID import faceID
from camera_to_image_array import camera_to_image_array as cam
from dataface_funcs import dataface_add
from getting_name_dataface import match_descriptor_in_dataface as lookup
from image_array_to_descriptors import match_image_to_descriptors as match
from dataface_funcs import pickOpen

def face_rec():
    pic = cam()
    dataface = pickOpen()
    out, rect = match(pic)
    names = lookup(dataface, out)
    faceID(pic, rect, names)
