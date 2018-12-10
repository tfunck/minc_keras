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
from utils import *
from custom_loss import *
from plot_metrics import *

def create_dir_verbose(directory):
    if not exists(directory): 
        makedirs(directory)
        print("Created directory:", directory)
    return directory
        
def setup_dirs(target_dir="./") :
    global data_dir
    global report_dir
    global train_dir
    global test_dir
    global validate_dir
    global model_dir
    data_dir = target_dir + os.sep + 'data'+os.sep
    report_dir = target_dir+os.sep+'report'+os.sep
    train_dir = target_dir+os.sep+'predict'+os.sep+'train'+os.sep
    test_dir = target_dir+os.sep+'predict'+os.sep+'test'+os.sep
    validate_dir = target_dir+os.sep+'predict'+os.sep+'validate'+os.sep
    model_dir=target_dir+os.sep+'model'
    create_dir_verbose(train_dir)
    create_dir_verbose(test_dir)
    create_dir_verbose(validate_dir)
    create_dir_verbose(data_dir)
    create_dir_verbose(report_dir) 
    create_dir_verbose(model_dir)    
    return 0
        
        
def minc_keras(source_dir, target_dir, input_str, label_str, ratios, feature_dim=2, batch_size=2, nb_epoch=10, images_to_predict=None, clobber=False, model_fn='model.hdf5',model_type='custom', images_fn='images.csv',nK="16,32,64,128", n_dil=None, kernel_size=3, drop_out=0, loss='categorical_crossentropy', activation_hidden="relu", activation_output="sigmoid", metric="categorical_accuracy", pad_base=0,  verbose=1, make_model_only=False ):
    
    setup_dirs(target_dir)

    images_fn = set_model_name(images_fn, report_dir, '.csv')
    [images, data] = prepare_data(source_dir, data_dir, report_dir, input_str, label_str, ratios, batch_size,feature_dim, images_fn,pad_base=pad_base,  clobber=clobber)

    ### 1) Define architecture of neural network
    Y_validate=np.load(data["validate_y_fn"]+'.npy')
    nlabels=len(np.unique(Y_validate))#Number of unique labels in the labeled images
    model = make_model(data["image_dim"], nlabels,nK, n_dil, kernel_size, drop_out, model_type, activation_hidden=activation_hidden, activation_output=activation_output)
    if make_model_only : return(0)

    ### 2) Train network on data
    model_fn =set_model_name(model_fn, model_dir)
    history_fn = splitext(model_fn)[0] + '_history.json'

    print( 'Model:', model_fn)
    if not exists(model_fn) or clobber:
    #If model_fn does not exist, or user wishes to write over (clobber) existing model
    #then train a new model and save it
        X_train=np.load(data["train_x_fn"]+'.npy')
        Y_train=np.load(data["train_y_fn"]+'.npy')
        X_validate=np.load(data["validate_x_fn"]+'.npy')
        model,history = compile_and_run(model, model_fn, history_fn, X_train,  Y_train, X_validate,  Y_validate, nb_epoch, nlabels, loss=loss, verbose=verbose)

    ### 3) Evaluate model on test data
    model = load_model(model_fn)
    X_test=np.load(data["test_x_fn"]+'.npy')
    Y_test=np.load(data["test_y_fn"]+'.npy')
    if loss in categorical_functions :
        Y_test=to_categorical(Y_test)
    test_score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test: Loss=', test_score[0], 'Metric=', test_score[1])
    #np.savetxt(report_dir+os.sep+'model_evaluate.csv', np.array(test_score) )

    ### 4) Produce prediction
    #predict(model_fn, validate_dir, data_dir, images_fn, images_to_predict=images_to_predict, category="validate", verbose=verbose)
    #predict(model_fn, train_dir, data_dir, images_fn, images_to_predict=images_to_predict, category="train", verbose=verbose)
    predict(model_fn, test_dir, data_dir, images_fn, loss, images_to_predict=images_to_predict, category="test", verbose=verbose)
    plot_loss(metric, history_fn, model_fn, report_dir)

    return 0



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='size of batch')
    parser.add_argument('--source', dest='source_dir', required=True, type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', required=True,type=str, default="results", help='target directory for output (Default: results)')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='number of training epochs')
    parser.add_argument('--pad', dest='pad', type=int,default=0, help='Images must be divisible by 2^<pad>. Default = 0 ')
    parser.add_argument('--loss', dest='loss', type=str,default='categorical_crossentropy', help='Loss function to optimize network')
    parser.add_argument('--nK', dest='nK', type=str,default='16,32,64,128', help='number of kernels')
    parser.add_argument('--n_dil', dest='n_dil', type=str,default=None, help='number of dilations')
    parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3, help='Size of kernels')
    parser.add_argument('--drop-out', dest='drop_out', type=float,default=0.0, help='Drop out rate')
    parser.add_argument('--metric', dest='metric', type=str,default='categorical_accuracy', help='Categorical accuracy')
    parser.add_argument('--activation-output', dest='activation_output', type=str,default='softmax', help='Activation function for last layer of network')
    parser.add_argument('--activation-hidden', dest='activation_hidden', type=str,default='relu', help='Activation function for core convolutional layers of network')
    
    #parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=2, help='Warning: option temporaily deactivated. Do not use. Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--model', dest='model_fn', default='model.hdf5',  help='model file where network weights will be saved/loaded. will be automatically generated if not provided by user')
    parser.add_argument('--model-type', dest='model_type', default='model_0_0',  help='Name of network architecture to use (Default=model_0_0): unet, model_0_0 (simple convolution-only network), dil (same as model_0_0 but with dilations).')
    parser.add_argument('--ratios', dest='ratios', nargs=2, type=float , default=[0.7,0.15,0.15],  help='List of ratios for training, validating, and testing (default = 0.7 0.15 0.15)')
    parser.add_argument('--predict', dest='images_to_predict', type=str, default=None, help='either 1) \'all\' to predict all images OR a comma separated list of index numbers of images on which to perform prediction (by default perform none). example \'1,4,10\' ')
    parser.add_argument('--input-str', dest='input_str', type=str, default='pet', help='String for input (X) images')
    parser.add_argument('--label-str', dest='label_str', type=str, default='brainmask', help='String for label (Y) images')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('--make-model-only', dest='make_model_only',  action='store_true', default=False,  help='Only build model and exit.')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,default=1, help='Level of verbosity (0=silent, 1=basic (default), 2=detailed, 3=debug')
    args = parser.parse_args()
    args.feature_dim =2

    minc_keras(args.source_dir, args.target_dir, input_str=args.input_str, label_str=args.label_str, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber, model_fn = args.model_fn ,model_type=args.model_type, images_to_predict= args.images_to_predict, loss=args.loss, nK=args.nK, n_dil=args.n_dil, kernel_size=args.kernel_size, drop_out=args.drop_out, activation_hidden=args.activation_hidden, activation_output=args.activation_output, metric=args.metric, pad_base=args.pad, verbose=args.verbose, make_model_only=args.make_model_only)
