
# coding: utf-8

# In[ ]:


class Node:
    """ Describes a node in a graph, and the edges connected
        to that node."""

    def __init__(self, ID, neighbors, dists, descriptor, truth=None, file_path=None):
        """ Parameters
            ----------
            ID : int
                A unique identifier for this node. Should be a
                value in [0, N-1], if there are N nodes in total.

            neighbors : Sequence[int]
                The node-IDs of the neighbors of this node.

            descriptor : numpy.ndarray
                The (128,) descriptor vector for this node's picture

            truth : Optional[str]
                If you have truth data, for checking your clustering algorithm,
                you can include the label to check your clusters at the end.

                If this node corresponds to a picture of Ryan, this truth
                value can just be "Ryan"

            file_path : Optional[str]
                The file path of the image corresponding to this node, so
                that you can sort the photos after you run your clustering
                algorithm

            """
        self.id = ID  # a unique identified for this node - this should never change

        # The node's label is initialized with the node's ID value at first,
        # this label is then updated during the whispers algorithm
        self.label = ID

        # (n1_ID, n2_ID, ...)
        # The IDs of this nodes neighbors. Empty if no neighbors
        self.neighbor_dist = tuple(neighbors)
        self.descriptor = descriptor

        self.truth = truth
        self.file_path = file_path


# In[1]:


import numpy as np
from scipy.spatial.distance import euclidean


# In[44]:


def populate_graph(image_descr, threshold=.5):
    """
    
    Given an array of image descriptors and a threshold that dictates whether two images might even be 
    connected, returns a populated graph with the structure 
    
    PARAMETERS
    ----------
    image_descr:   np.ndarray (N,128) descriptors of each of the images
    threshold:    int, how to absolutely declare two pictures different
    
    RETURN
    -----
    output:   array of tuples (M,) with each tuple containing the clustering of images of the same person
    
    """
    
    a = list()
    for i in range(image_descr.shape[0]):
        a.append(list())
    for i,im in enumerate(image_descr):
        for j,img in enumerate(image_descr):
            if i==j:
                continue
            d = euclidean(img,im)
            if d < threshold:
                a[i].append(tuple((j,d))) #tuples are (im_id, distance)
                a[j].append(tuple((i,d)))
    return a
    


# In[45]:


im = np.array(((5,4,3),(0,0,1),(2,3,4),(1,0,1)))

clustering(im, threshold=10)

