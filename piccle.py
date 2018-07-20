from pickle import load, dump

def pickOpen(path = 'dataface.dat'):
    file = open(path, 'rb')
    return load(file)

def pickSave(obj, path = 'dataface.dat'):
    file = open(path, 'wb')
    dump(obj, file)
