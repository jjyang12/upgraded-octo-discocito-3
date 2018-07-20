
import numpy as np
from scipy.spatial.distance import euclidean


def match_descriptor_in_dataface(dataface, descrip_array, threshold=.4):
    
    """
    Given dictionary and a list of face descriptors, returns an array with names corresponding with descrip_array input
    
    PARAMETERS
    ----------
    dataface:  dictionary with the name as the key, mapping to their face descriptor
    descrip_array:  np.array (N,) with the descriptors to be matched
    
    RETURNS
    ------
    labels: np.ndarray (N,) with names of people in the image, matching the indexes of the input. 
        If distance is above threshold, returns -1"""
    
    #print(descrip_array.shape[0])
    labels = ["" for x in range(descrip_array.shape[0])]
    for i,d in enumerate(descrip_array):
        min_dist = 100
        face = str(0)
        for key in dataface:
            #print (dataface[key])
            #print(d.shape)
            dist = euclidean(dataface[key],d)
            if dist < min_dist:
                min_dist = dist
                labels[i] = key
        if min_dist > threshold:
            labels[i] = 'Unknown'
              
    return labels
        


# In[30]:

"""
data = {'jenny': (np.array(([1,3,4]))), 'sam': (np.array(([0,1,2])))}
arr = np.array([[500,100,3],[0,1,2]])

match_descriptor_in_dataface(data, arr)


# In[ ]:
"""


    

