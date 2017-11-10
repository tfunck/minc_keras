import os
import h5py
import numpy as np
from os.path import splitext, basename, exists

def set_model_name(filename, target_dir, ext='.hdf5'):
    '''function to set default model name'''
    return  target_dir+os.sep+splitext(basename(filename))[0]+ext


def safe_h5py_open(filename, mode):
    '''open hdf5 file, exit elegantly on failure'''
    try :
        f = h5py.File(filename, mode)
        return f
    except OSError :
        print('Error: Could not open', filename)
        exit(1)


def normalize(A):
    '''performs a simple normalization from 0 to 1 of a numpy array. checks that the image is not a uniform value first

    args
        A -- numpy array

    returns
        numpy array (either A or normalized version of A)
    '''
    A = (A - np.mean(A)) / np.std(A)

    scale_factor=(np.max(A)-np.min(A))
    if scale_factor==0: return A

    return (A - np.min(A))/scale_factor




'''# fix random seed for reproducibility
np.random.seed(8)
def generator(f, batch_size):
    i=0
    start=i*batch_size
    end=start + batch_size
    #while end < tensor_max:
    while True:
        start=i*batch_size
        end=start + batch_size #(i+1)*batch_size 
        X = f['image'][start:end,]
        Y = f['label'][start:end,]
        i+=1
        yield [X,Y]
       '''
