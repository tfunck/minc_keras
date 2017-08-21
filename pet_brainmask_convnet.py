import numpy as np
import scipy as sp
import pandas as pd
from pyminc.volumes.factory import *
import os
from re import sub
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.preprocessing.image import ImageDataGenerator
from sys import argv, exit
from glob import glob
from os.path import basename, exists
# fix random seed for reproducibility
np.random.seed(8)

def set_images(subject_dirs):
    '''Creates a DataFrame that contains a list of all the subjects along with their PET images, T1 MRI images and labeled images.'''

    subject_dirs = glob(source_dir+os.sep+'*')
    pet_list = glob( source_dir + os.sep + '*' + os.sep + '*_pet.*'  )
    t1_list = glob( source_dir + os.sep + '*' + os.sep + '*_t1.*'  )
    label_list = glob( source_dir + os.sep + '*' + os.sep + '*_labels.*'  )
    names = [ basename(f) for f in  subject_dirs ]
    colnames=["subject", "radiotracer", "pet", "t1", "label"]
    out=pd.DataFrame(columns=colnames)
    for name in names:
        label = [ f for f in label_list if name in f][0] 
        pet =  [ f for f in pet_list if name in f ]
        t1 =  [ f for f in t1_list if name in f ][0]
        pet_names = [ sub('acq-','', g) for f in pet for g in f.split('_') if 'acq' in g ]
        n=len(pet)
        subject_df = pd.DataFrame(np.array([[name] * n,  pet_names,pet,[t1]*n, [label] * n]).T, columns=colnames)
        out = pd.concat([out, subject_df ])
    return out

def feature_extraction(images, target_dir, chunks=1, feature_type=1, parameters=None, normalize=False, clobber=False):
    '''Extracts the features from the PET images according to option set in feature type.
    Feature type options: 
        1) Full image (no features extracted): return 3d array
            Parameters = None
        2) Slice: return list of 2d image slices 
            Parameters = integer, axis (0,1,2) along which to extract slices (default=0=z=axial)
        3) Lines: return list of 1d profiles 
            Parameters = None
        4) 2D Kernel: return list of 2d patches from full image
            Parameters = integer, size of kernel 
        5) 3D Kernel: return list of 3d patches from full image
            Parameters = integer, size of kernel 
    '''
    n=len(chunks)
    m=images.shape[0]
    temp=target_dir+os.sep+'chunks'
    if not exists(temp): os.makedirs(temp)

    X_csv=temp +os.sep+ 'X_type-'+str(feature_type)+'_chunk-'+str(chunks[0])+'-'+str(n)+'-'+str(m)
    Y_csv=temp +os.sep+ 'Y_type-'+str(feature_type)+'_chunk-'+str(chunks[0])+'-'+str(n)+'-'+str(m)
    if not exists(X_csv+'.npy') or not exists(Y_csv+'.npy') or clobber==True :
        for i in chunks:
            j=i-min(chunks)
            row=images.iloc[i, ] 

            pet=volumeFromFile(row.pet).data
            label=volumeFromFile(row.label).data
            if len(pet.shape) == 4: pet = np.sum(pet, axis=0)
            if normalize==True: pet / pet.max()

            try : X
            except NameError : X=np.zeros([n]+list(pet.shape))

            try : Y
            except : Y=np.zeros([n]+list(label.shape))
            X[j]=pet 
            Y[j]=label
            
        np.save(X_csv, X)
        np.save(Y_csv, Y)
    else:
        X=np.load(X_csv+'.npy')
        Y=np.load(Y_csv+'.npy')
        
    return [X, Y] 


#def define_arch():
    '''Define architecture of neural net'''
    # create model
#    model = Sequential()
#    model.add(Dense(12, input_dim=8, activation='relu'))
#    model.add(Dense(8, activation='relu'))
#    model.add(Dense(1, activation='sigmoid'))
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#    return model

def pet_brainmask_convnet(source_dir, target_dir, chunk_size=2,nb_epoch=1, clobber=False ):
    #1) Organize inputs into a data frame, match each PET image with label image
    images = set_images(source_dir)
    #2) Extract the features we want from input images
    #3) Define the inputs and outputs to system archetecture
    #model = define_arch()

    #model.fit(X, Y, epochs=150, batch_size=10)

    #scores = model.evaluate(X, Y)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    n = images.shape[0]
    print(images.shape)
    nchunks = n / chunk_size
    for e in range(nb_epoch):
        print("epoch %d" % e)
        for i in np.arange(0, n, chunk_size, dtype=int):
            chunks = np.arange(i, i + chunk_size)
            chunks = chunks[ chunks < n ]
            #print( chunks)
            X_train, Y_train = feature_extraction(images,  target_dir, chunks, normalize=True, clobber=clobber ) 
            
            #if X_train == [] or Y_train == []: break
            #model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
    

    return 0


if __name__ == '__main__':
    source_dir = argv[1]
    target_dir = argv[2]
    chunk_size = int(argv[3])
    nb_epoch = int(argv[4])
    pet_brainmask_convnet(source_dir, target_dir, chunk_size, nb_epoch, clobber=True)
