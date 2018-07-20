def dataface_add(dataface, name, descriptor, amount=0.25):
    '''
    Logs the given entry (name, descriptor) in the dataface
    If an entry with the given name does not exist, enter the value directly
    Otherwise, modify the existing value according to amount
    
    INPUTS:
    
        dataface:   A dictionary of names to ndarray descriptors
                    dict
        
        name:       Name of the person to add
                    string
        
        descriptor: Face descriptor returned by the .compute_face_descriptor() method of dlib's "face rec" model
                    ndarray shape (128,)
        
        amount:     Amount to modify the descriptor by if it exists under the given name
                    0.0 means do not modify, 1.0 means replace entirely
    
    OUTPUT:
    
        None    
    '''
    if name not in dataface or amount == 1.0:
        # replace entirely
        dataface[name] = descriptor
    elif amount != 0.0:
        # modify partially
        old = dataface[name]
        dataface[name] = (1-amount) * old + amount * descriptor

from pickle import load, dump

def pickOpen(path = '../dataface.dat'):
    file = open(path, 'rb')
    return load(file)

def pickSave(obj, path = '../dataface.dat'):
    file = open(path, 'wb')
    dump(obj, file)
