import numpy as np
#from pyminc.volumes.factory import *
import os
from sys import argv, exit
from os.path import  exists
from os import makedirs
import argparse
#local modules defined in current project
from make_and_run_model import *
from predict import *
from prepare_data import *

# fix random seed for reproducibility
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
       


def set_model_name(target_dir, feature_dim):
    '''function to set default model name'''
    model_dir=target_dir+os.sep+'model'
    if not exists(model_dir): makedirs(model_dir)
    return model_dir+os.sep+ 'model_'+str(feature_dim)+'.hdf5'

def pet_brainmask_convnet(source_dir, target_dir, input_str, label_str, ratios, feature_dim=2, batch_size=2, nb_epoch=10, images_to_predict=None, clobber=False, model_name=False, verbose=1 ):
    images = prepare_data(source_dir, target_dir, input_str, label_str, ratios, batch_size,feature_dim, clobber)
    ### 1) Define architecture of neural network
    model = make_model(batch_size)

    ### 2) Train network on data

    if model_name == None:  model_name =set_model_name(target_dir, feature_dim) 
    if not exists(model_name) or clobber:
    #If model_name does not exist, or user wishes to write over (clobber) existing model
    #then train a new model and save it
        X_train=np.load(prepare_data.train_x_fn+'.npy')
        Y_train=np.load(prepare_data.train_y_fn+'.npy')
        X_test=np.load(prepare_data.test_x_fn+'.npy')
        Y_test=np.load(prepare_data.test_y_fn+'.npy')
        model = compile_and_run(model, X_train, Y_train, X_test, Y_test, prepare_data.batch_size, nb_epoch)
        model.save(model_name)

    ### 3) Produce prediction
    predict(model_name, target_dir, images, images_to_predict, verbose)

    return 0



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, help='target directory')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='number of training epochs')
    parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=2, help='Warning: option temporaily deactivated. Do not use. Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--model', dest='model_name', default=None,  help='model file where network weights will be saved/loaded. will be automatically generated if not provided by user')
    parser.add_argument('--ratios', dest='ratios', nargs=2, type=float , default=[0.7,0.3],  help='List of ratios for training, testing, and validating (default = 0.7 0.3')
    parser.add_argument('--predict', dest='predict', action='store_true', default=False, help='perform prediction only (assumes model file exists) ')
    parser.add_argument('--images-to-predict', dest='images_to_predict', type=str, default=None, help='either 1) \'all\' to predict all images OR a comma separated list of index numbers of images on which to perform prediction (by default perform none). example \'1,4,10\' ')
    parser.add_argument('--input-str', dest='input_str', type=str, default='pet', help='String for input (X) images')
    parser.add_argument('--label-str', dest='label_str', type=str, default='brainmask', help='String for label (Y) images')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,default=1, help='Level of verbosity (0=silent, 1=basic (default), 2=detailed, 3=debug')
    args = parser.parse_args()
    args.feature_dim =2

    pet_brainmask_convnet(args.source_dir, args.target_dir, input_str=args.input_str, label_str=args.label_str, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber, model_name = args.model_name ,images_to_predict= args.images_to_predict, verbose=args.verbose)
