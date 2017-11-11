import numpy as np
import pandas as pd
from re import sub
from sys import argv, exit
from glob import glob
import os
from os.path import basename, exists
from os import makedirs
import time
from utils import *
np.random.seed(int(time.time()))

def gather_dirs(source_dir, input_str='acq', ext='mnc'):
    ''' This function takes a source directory and parse it to extract the
        information and returns it as lists of strings.

    Args:
        source_dir (str): the directory where the data is
        input_str (str): string that uniquely identifies the input (pet) images in subject directory
        ext (str) : string the identifies extension of file (default==nifti)

    Returns:
        subject_dir (list of str): a list of directories, one per subject
        pet_list (list of str): a list of directories, one per pet image
        t1_list (list of str): a list of directories, one per t1 image
        names (list of str): a list of subject names
    '''
    input_str = os.path.splitext(input_str)[0]
    subject_dirs = glob(source_dir + os.sep + '*', recursive=True)
    pet_list = glob(source_dir + os.sep + '**' + os.sep + '*'+input_str+'.'+ext, recursive=True)
    if len(pet_list) == 0 : print('Warning: could not find input file of form:', source_dir + os.sep + '**' + os.sep + '*_'+input_str+'.'+ext)
    names = [basename(f) for f in subject_dirs]
    #return(subject_dirs, pet_list, t1_list, names)
    return(subject_dirs, pet_list, names)


def print_error_nosubject(source_dir):
    ''' This function displays an error message if the source directory
        is empty, then exits the execution.

    Args:
        source_dir (str): the directory where the data is (not)

    '''
    print('No data found in directory :' + source_dir)
    print('Are you sure this is the right directory ?')
    print('Exiting gracefully')
    exit()


def print_error_nosumone(ratio):
    ''' This function displays an error message if the ratio sum
        is not one.

    Args:
        ratio (list): a list containing the proportions of train/test
            subjects. should sum to 1.


    '''
    print('The sum of ' + str(ratio) + ' is not one.')
    print('Check again your test/train ratios')
    print('Exiting gracefully')
    exit()


#def createdf(name, pet_names, pet, t1, labels, task=False):
def createdf(name, pet_names, pet, labels, task=False):
    ''' This function creates a  dataframe for a given subject.
        The dataframe contains information about the files related to
        the given subject.

    Args:
        name (str) : the name of the subject
            (example: 'subject-001')
        pet_names (list) : a list of the radiotracer used
            (example: [raclopride, FDG])
        pet (list): a list of (mnc) file names where pet information is stored.
            (example: ['../data/sub-02/sub-02_acq-raclopride_pet.mnc',
             '../data/sub-02/sub-02_acq-FDG_pet.mnc'] )
        t1 (str): a (mnc) file name where the t1 information is stored
            (example: '../data/sub-02/sub-02_t1.mnc')
        labels (list): a list of str of (mnc) file name where the label is.
            (example: ['../data/sub-02/sub-02_labels_brainmask.mnc'])
        task (bool or list of str): leave it to False if your data has no task

    Returns:
        datad (pd.DataFrame): a dataframe that has all these informations
            in the right format and number of repetition by subject.
    '''

    n = len(pet)
    datad = {}
    datad['subject'] = [name] * n
    if pet_names != []: datad['radiotracer'] = pet_names
    datad['pet'] = pet
    #datad['t1'] = [t1] * n

    if task is not False:
        datad['label'] = labels
        datad['task'] = task
    else:
        datad['label'] = labels * n
    return(datad)


def process(name, source_dir, pet_list,  label_str='brainmask', ext='mnc' ):
#def process(name, source_dir, pet_list, t1_list, label_str='brainmask', ext='mnc' ):
    ''' That function returns a dataframe that has all the information
        about a subject. At this point of the code, the category of
        a subject (train/test) is still unknown. Given the presence or
        absence of 'tasks', the returned dataframe has 5 or 6 columns.

    Args:
        name(str): the name of the subject
        source_dir(str): the directory where the data is
        pet_list(list): a list of (mnc) file names where pet
            information is stored.
        t1_list (list): a list of file names. these files are all mnc t1s.

    Returns:
        data_subject (pd.DataFrame): a dataframe that has all the information
            about a subject. The number of columns depends on the presence of
            tasks (ie augmented data).


    '''
    pet = [f for f in pet_list if name in f]
    
    #t1 = [f for f in t1_list if name in f]
    #if not t1 == [] : t1=t1[0]
    #else: 
    #    print('Warning: Subject name '+name+' not found in list of t1 images.')
    #    return(1)
    pet_names = [sub('.mnc', '', sub('acq-', '', g))
                 for f in pet for g in f.split('_') if 'acq' in g]
    task_names = [sub('task-', '', g)
                  for f in pet for g in f.split('_') if 'task' in g]

    label_str = os.path.splitext(label_str)[0]
    if len(task_names) == 0:
        label = glob(source_dir + os.sep + '**' +  os.sep + '*'+label_str+'.'+ext, recursive=True)
        data_subject = createdf(name, pet_names,  pet, label, task=False)

    else :
        labels = []
        for p, t in zip(pet, task_names):
            label_fn = glob(source_dir + os.sep + '**' + os.sep + name + '*' + t + '*'+label_str+'.'+ext, recursive=True)
            if not label_fn == []: label_fn = label_fn[0]
            else: 
                print('Warning: could not find label for ', name, 'with the form:')
                print(source_dir + os.sep + '**' + os.sep + name + '*' + t + '*'+label_str+'.'+ext)
                return(1)
            
            labels.append(label_fn)
        #data_subject = createdf(name, pet_names,pet, t1, labels,task=task_names)
        data_subject = createdf(name, pet_names,pet, labels,task=task_names)
    return(data_subject)


def create_out(dfd):
    ''' Function that goes from a dict of df to a single df (successive
        concatenation).

    Args :
        dfd (dict): a dictionary of pd.DataFrame

    Returns:
        out (pd.DataFrame): a pd.DataFrame that is a concatenation of all the
            pd.DataFrames in dfd. It also has an extra 'category' column and
            the index is reset.

    '''
    if len(list(dfd.keys())) == 0 :
        print('Error: subject data frame was empty')
        exit(1)

    out = pd.DataFrame(columns=dfd[list(dfd.keys())[0]].columns)
    for k, v in dfd.items():
        out = pd.concat([out, v])
    out["category"] = "unknown"
    out.reset_index(inplace=True, drop=True)
    return(out)

def attribute_category(out, category, ratio,  verbose=1):
    ''' This function distributes each subject in a 'train' or 'test' category. The 'train' and 'test' 
        categories are assigned so as to make sure that all of the different radiotracers are contained 
        within the 'train' category.

    Args:
        out (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject.
        ratios (list): a list containing the proportions of train/test
            subjects. should sum to 1 and supposedly it has been tested before.

    Returns:
        out (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject where the 'category' column has been set to either
            train or test depending the result of the random draw.
            The value of test or train is the same for a given subject.
    '''
    nImages=out.shape[0]
    n = int(round(nImages * ratio))
    i=0

    radiotracers = pd.Series(out.radiotracer)
    unique_radiotracers = np.unique(radiotracers)
    while True : 
        for r in unique_radiotracers :
            unknown_df = out[ (out.category == "unknown") & (radiotracers == r) ]
            n_unknown = unknown_df.shape[0]
            if n_unknown == 0: continue
            random_i = np.random.randint(0,n_unknown)
            row = out[out.category == "unknown"].iloc[random_i,]
            out.loc[ out.index[out.subject == row.subject ], 'category'  ]=category
            i +=  out.loc[ out.index[out.subject == row.subject ], 'category'  ].shape[0]
            if i >= n : break

        n_unknown = out[ (out.category == "unknown") & (radiotracers == r) ].shape[0]
        if i >= n or n_unknown == 0  : break
    
    if verbose > 0 : 
        print(category, ": expected/real ratio = %3.2f / %3.2f" % (100. * ratio, 100.*out.category.loc[ out.category ==category].shape[0]/nImages ))



def set_valid_samples(images):
    '''for each image, identify the number of samples that are valid. some samples should be excluded because they contain
    bad or no information
    
    '''
    total_slices=0
    images['valid_samples']=np.repeat(0, images.shape[0])
    images['total_samples']=np.repeat(0, images.shape[0])
    for index, row in images.iterrows():
        minc_pet_f = safe_h5py_open(row.pet, 'r')
        pet=np.array(minc_pet_f['minc-2.0/']['image']['0']['image'])
        if len(pet.shape) == 4: pet = np.sum(pet, axis=0)

        images['total_samples'].iloc[index] = pet.shape[0]
        valid_slices = 0
        for j in range(pet.shape[0]):
            if pet[j].sum() != 0 : 
                valid_slices += 1
        images['valid_samples'].iloc[index] = valid_slices
        total_slices += valid_slices
    return(total_slices)

def set_images(source_dir, target_dir, ratios, images_fn, input_str='pet', label_str='brainmask', ext='mnc' ):
    ''' This function takes a source directory, where the data is, and a
        ratio list (split test/train).
        It returns a pd.DataFrame that links file names to concepts, like
        t1 of subject 2 or pet-FDG for subject 15.
        This dataframe is exported is csv.

        Args:
            source_dir (str): the directory where the data is
            target_dir (str): the directory where the results will go
            input_str  (str): string used to identify input files
            label_str  (str): string used to identify label files
            ext (str) : string the identifies extension of file (default==nifti)
        Returns:
            out (pd.DataFrame): a dataframe that synthesises the information
                of the source_dir.

    '''

    # 1 - gathering information (parsing the source directory)
    subject_dirs, pet_list, names = gather_dirs(source_dir, input_str, ext )
    
    # 2 - checking for potential errors
    if len(names) == 0:
        print_error_nosubject(source_dir)
    if sum(ratios) != 1:
        print_error_nosumone(ratios)

    # 3 - creating an empty directory of dataframes, then filling it.
    dfd = {}
    for name in names:
        data_subject = process(name, source_dir, pet_list, label_str, ext)
        #data_subject = process(name, source_dir, pet_list, t1_list, label_str, ext)
        if not data_subject == 1: dfd[name] = pd.DataFrame(data_subject)  # formerly subject_df
    # 4 - concatenation of the dict of df to a single df
    out = create_out(dfd)

    ## 4.5) create one hot label for pet images
    unique_radiotracers = dict( enumerate( out.radiotracer.unique() ) )
    unique_one_hot =  dict([ (item, key) for key, item in unique_radiotracers.items() ]) 
    out["onehot"] = [ unique_one_hot[i] for i in out.radiotracer ] 
    
    
    # 5 - attributing a train/validate/test category for all subject
    attribute_category(out, 'train', ratios[0])
    attribute_category(out, 'validate', ratios[1])
    out.category.loc[ out.category=="unknown" ] = "test"
    #5.5 Set the number of valid samples per image (some samples exluded because they contain no information)
    set_valid_samples(out)

    # 6 - export and return
    out.to_csv(images_fn, index=False)
    return out


if __name__ == '__main__':
    set_images(argv[1], [0.7, 0.3])
