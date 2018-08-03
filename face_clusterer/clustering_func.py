from scipy.spatial.distance import euclidean

def populate_graph(image_descr, truth, threshold=.5):
    """
    
    Given an array of image descriptors and a threshold that dictates whether two images might even be 
    connected, returns a populated graph with the structure 
    
    PARAMETERS
    ----------
    image_descr:   np.ndarray (N, 128) descriptors of each of the images
    threshold:     float, how to absolutely declare two pictures different
    truth:         array containing the groundtruth values of each node
    
    RETURN
    -----
    graph:  The created graph
            tuple of (edges, truth, predicted) where
                edges is an array size N of arrays of tuples representing the connections at each node
                    subarrays contain tuples of (id, distance)
                truth is an array size N containing the groundtruth values of each node
                    literally the same thing as the passed in truth parameter
                predicted is an array size N containing values predicted by clustering for each node
                    here it's returned with every label equal to that node's id
    """
    
    edges = list()
    for i in range(image_descr.shape[0]):
        edges.append(list())
    for i,im in enumerate(image_descr):
        for j,img in enumerate(image_descr):
            if i==j:
                continue
            d = euclidean(img,im)
            if d < threshold:
                edges[i].append(tuple((j,d))) #tuples are (id, distance)
                edges[j].append(tuple((i,d)))
    return (edges, truth, list(i for i in range(image_descr.shape[0])))
