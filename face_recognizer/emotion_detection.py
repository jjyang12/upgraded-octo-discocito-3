import pickle
import numpy as np
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.activations.relu import relu

from matplotlib.patches import Rectangle
from matplotlib.pyplot import text
import pylab as plt
from mynn.activations.softmax import softmax

def create_model(path="emotion_net_parameters.dat"):
    '''
    Creates the emotion detection model using trained parameters.
    
    PARAMETERS:
        path : string
            the path to the emotion neural net parameters
    
    OUTPUT:
        An instance of a trained emotion detection model.
    '''
    
    with open(path, mode="rb") as f:
        new_parameters = pickle.load(f)

    class Model:
        def __init__(self):
            self.dense1 = dense(2*68, 100, weight_initializer=glorot_normal)
            self.dense1.weight = new_parameters[0]
            self.dense1.bias = new_parameters[1]
            self.dense2 = dense(100, 10, weight_initializer=glorot_normal)
            self.dense2.weight = new_parameters[2]
            self.dense2.bias = new_parameters[3]
            self.dense3 = dense(10, 3, weight_initializer=glorot_normal)
            self.dense3.weight = new_parameters[4]
            self.dense3.bias = new_parameters[5]

        def __call__(self, x):
            return self.dense3(relu(self.dense2(relu(self.dense1(x)))))

        @property
        def parameters(self):
            return self.dense1.parameters + self.dense2.parameters + self.dense3.parameters
       
    return Model()

def emotionID(pic, ax, face_detect, shape_predictor, emotion_detect):
    '''
    Displays emotions located around the face.
    
    PARAMETERS:
        pic - The image array with the face to be modified.
        ax - The plot's ax.
        face_detect - The face detection model.
        shape_predictor - The shape predictor model.
        emotion_detect - The emotion detection model.
        
    OUTPUT:
        emotion - The detected emotion.
    '''
    
    detections = list(face_detect(pic))
    assert len(detections) == 1, 'Only 1 face is supported at this time.'
    
    new_labels = ["sad", "neutral", "happy"]
    
    landmarks = shape_predictor(pic, detections[0])
    landmarks_arr = np.empty((68, 2))
    for i in range(68):
        landmarks_arr[i, 0] = landmarks.part(i).x
        landmarks_arr[i, 1] = landmarks.part(i).y

    mean = np.mean(landmarks_arr, axis=0)
    std = np.std(landmarks_arr, axis=0)
    landmarks_norm = (landmarks_arr - mean) / std
    
    # try to detect the emotion
    landmarks_final = landmarks_norm.reshape(68*2)
    probs = emotion_detect(landmarks_final)
    probs_soft = softmax(probs)
    result = {emotion:probs_soft.data[0][i] for i, emotion in enumerate(new_labels)}
    
    top_emotion = max(result.keys(), key=(lambda k: result[k]))
    
    l, t, r, b = detections[0].left(), detections[0].top(), detections[0].right(), detections[0].bottom()
    rect = Rectangle((l,b) , r - l, t - b, fill = False)
    ax.add_patch(rect)

    loc = text(l, b + 40, top_emotion, bbox = dict(facecolor='white', alpha=1))
    return top_emotion
