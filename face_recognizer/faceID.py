from matplotlib.patches import Rectangle
from matplotlib.pyplot import text
import pylab as plt


def faceID(pic, detections, names):
    '''
    Displays the picture with boxes and names located around each face.
    
    PARAMETERS:
        pic - The image array with the faces to be modified.
        detections - The face detection data. Output of the face_detect function in list form.
        names - The list of names associated with the faces. Indexing should correlate with detections.
        
    OUTPUT:
        ax - The plot's ax, so that an emotion ID can also be plotted.
    '''
    
    assert len(detections) == len(names), 'Names list must correlate with detections list.'
    
    fig, ax = plt.subplots()
    ax.imshow(pic)
    
    for k, d in enumerate(detections):
        # Get the landmarks/parts for the face in box d.
        # Draw the face landmarks on the screen.
    
        l, t, r, b = detections[k].left(), detections[k].top(), detections[k].right(), detections[k].bottom()
        rect = Rectangle((l,b) , r - l, t - b, fill = False)
        ax.add_patch(rect)
    
        loc = text(l, b - 20, names[k], bbox = dict(facecolor='white', alpha=1))
        
    return ax
