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