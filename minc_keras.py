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



def minc_keras(source_dir, target_dir, input_str, label_str, ratios, feature_dim=2, batch_size=2, nb_epoch=10, images_to_predict=None, clobber=False, model_fn='model.hdf5',model_type='model_0_0', images_fn='images.csv', loss='categorical_crossentropy', activation_hidden="relu", activation_output="sigmoid", metric="categorical_accuracy",  verbose=1 ):

    data_dir = target_dir + os.sep + 'data'+os.sep
    report_dir = target_dir+os.sep+'report'+os.sep
    train_dir = target_dir+os.sep+'predict'+os.sep+'train'+os.sep
    test_dir = target_dir+os.sep+'predict'+os.sep+'test'+os.sep
    validate_dir = target_dir+os.sep+'predict'+os.sep+'validate'+os.sep
    model_dir=target_dir+os.sep+'model'
    if not exists(train_dir): makedirs(train_dir)
    if not exists(test_dir): makedirs(test_dir)
    if not exists(validate_dir): makedirs(validate_dir)
    if not exists(data_dir): makedirs(data_dir)
    if not exists(report_dir): makedirs(report_dir) 
    if not exists(model_dir): makedirs(model_dir) 

    images_fn = set_model_name(images_fn, report_dir, '.csv')
    [images, image_dim] = prepare_data(source_dir, data_dir, report_dir, input_str, label_str, ratios, batch_size,feature_dim, images_fn,  clobber=clobber)

    ### 1) Define architecture of neural network
    Y_validate=np.load(prepare_data.validate_y_fn+'.npy')
    nlabels=len(np.unique(Y_validate))#Number of unique labels in the labeled images
    model = make_model(image_dim, nlabels, model_type, activation_hidden=activation_hidden, activation_output=activation_output)

    ### 2) Train network on data

    model_fn =set_model_name(model_fn, model_dir)
    history_fn = splitext(model_fn)[0] + '_history.json'

    print( 'Model:', model_fn)
    if not exists(model_fn) or clobber:
    #If model_fn does not exist, or user wishes to write over (clobber) existing model
    #then train a new model and save it
        X_train=np.load(prepare_data.train_x_fn+'.npy')
        Y_train=np.load(prepare_data.train_y_fn+'.npy')
        X_validate=np.load(prepare_data.validate_x_fn+'.npy')
        model,history = compile_and_run(model, model_fn, history_fn, X_train,  Y_train, X_validate,  Y_validate, nb_epoch, nlabels, loss=loss)

    ### 3) Evaluate model on test data
    model = load_model(model_fn)
    X_test=np.load(prepare_data.test_x_fn+'.npy')
    Y_test=np.load(prepare_data.test_y_fn+'.npy')
    #test_score = model.evaluate(X_test, Y_test, verbose=1)
    #print('Test: Loss=', test_score[0], 'Dice:', test_score[1])
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
    parser.add_argument('--source', dest='source_dir', type=str, help='source directory')
    parser.add_argument('--target', dest='target_dir', type=str, default="results", help='target directory for output (Default: results)')
    parser.add_argument('--epochs', dest='nb_epoch', type=int,default=10, help='number of training epochs')
    parser.add_argument('--loss', dest='loss', type=str,default='categorical_crossentropy', help='Loss function to optimize network')
    parser.add_argument('--metric', dest='metric', type=str,default='categorical_accuracy', help='Categorical accuracy')
    parser.add_argument('--activation-output', dest='activation_output', type=str,default='sigmoid', help='Activation function for last layer of network')
    parser.add_argument('--activation-hidden', dest='activation_hidden', type=str,default='relu', help='Activation function for core convolutional layers of network')
    
    #parser.add_argument('--feature-dim', dest='feature_dim', type=int,default=2, help='Warning: option temporaily deactivated. Do not use. Format of features to use (3=Volume, 2=Slice, 1=profile')
    parser.add_argument('--model', dest='model_fn', default='model.hdf5',  help='model file where network weights will be saved/loaded. will be automatically generated if not provided by user')
    parser.add_argument('--model-type', dest='model_type', default='model_0_0',  help='Name of network architecture to use (Default=model_0_0): unet, model_0_0 (simple convolution-only network), dil (same as model_0_0 but with dilations).')
    parser.add_argument('--ratios', dest='ratios', nargs=2, type=float , default=[0.7,0.15,0.15],  help='List of ratios for training, validating, and testing (default = 0.7 0.15 0.15)')
    parser.add_argument('--predict', dest='images_to_predict', type=str, default=None, help='either 1) \'all\' to predict all images OR a comma separated list of index numbers of images on which to perform prediction (by default perform none). example \'1,4,10\' ')
    parser.add_argument('--input-str', dest='input_str', type=str, default='pet', help='String for input (X) images')
    parser.add_argument('--label-str', dest='label_str', type=str, default='brainmask', help='String for label (Y) images')
    parser.add_argument('--clobber', dest='clobber',  action='store_true', default=False,  help='clobber')
    parser.add_argument('-v', '--verbose', dest='verbose', type=int,default=1, help='Level of verbosity (0=silent, 1=basic (default), 2=detailed, 3=debug')
    args = parser.parse_args()
    args.feature_dim =2

    minc_keras(args.source_dir, args.target_dir, input_str=args.input_str, label_str=args.label_str, ratios=args.ratios, batch_size=args.batch_size, nb_epoch=args.nb_epoch, clobber=args.clobber, model_fn = args.model_fn ,model_type=args.model_type, images_to_predict= args.images_to_predict, loss=args.loss, activation_hidden=args.activation_hidden, activation_output=args.activation_output, metric=args.metric, verbose=args.verbose)
