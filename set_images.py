import numpy as np
import pandas as pd
from re import sub
from sys import argv, exit
from glob import glob
import os
from os.path import basename, exists
from os import makedirs

def gather_dirs(source_dir, input_str='acq'):
    ''' This function takes a source directory and parse it to extract the
        information and returns it as lists of strings.

    Args:
        source_dir (str): the directory where the data is

    Returns:
        subject_dir (list of str): a list of directories, one per subject
        pet_list (list of str): a list of directories, one per pet image
        t1_list (list of str): a list of directories, one per t1 image
        names (list of str): a list of subject names
    '''
    subject_dirs = glob(source_dir + os.sep + '*')
    pet_list = glob(source_dir + os.sep + '*' + os.sep + '*_'+input_str+'*')
    t1_list = glob(source_dir + os.sep + '*' + os.sep + '*_t1.*')
    names = [basename(f) for f in subject_dirs]
    return(subject_dirs, pet_list, t1_list, names)


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


def createdf(name, pet_names, pet, t1, labels, task=False):
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
    datad['t1'] = [t1] * n

    if task is not False:
        datad['label'] = labels
        datad['task'] = task
    else:
        datad['label'] = labels * n
    return(datad)


def process(name, source_dir, pet_list, t1_list, label_str='brainmask'):
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
    t1 = [f for f in t1_list if name in f][0]
    pet_names = [sub('.mnc', '', sub('acq-', '', g))
                 for f in pet for g in f.split('_') if 'acq' in g]
    task_names = [sub('task-', '', g)
                  for f in pet for g in f.split('_') if 'task' in g]



    if len(task_names) == 0:
        label = glob(source_dir + os.sep + name +
                     os.sep + '*_labels_'+label_str+'.*')
        data_subject = createdf(name, pet_names,
                                pet, t1, label,
                                task=False)

    else :
        labels = []
        for p, t in zip(pet, task_names):
            label_fn = glob(source_dir + os.sep + '*' + os.sep +
                            name + '*' + t + '*'+label_str+'.*')[0]
            labels.append(label_fn)
        data_subject = createdf(name, pet_names,
                                pet, t1, labels,
                                task=task_names)
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
    out = pd.DataFrame(columns=dfd[list(dfd.keys())[0]].columns)
    for k, v in dfd.items():
        out = pd.concat([out, v])
    out["category"] = "unknown"
    out.reset_index(inplace=True, drop=True)
    return(out)


def attribute_category(out, ratios):
    ''' This function distributes each subject in a 'train' or 'test' category.

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
    nSubjects = len(out.subject.unique())
    i_train = np.random.choice(
        np.arange(nSubjects), int(ratios[0] * nSubjects))
    train_or_test_by_subject = [
        'train' if i in i_train else 'test' for i in range(nSubjects)]
    images_per_subject = out.groupby(["subject"]).category.count().values
    out.category = list(np.repeat(train_or_test_by_subject,
                                  images_per_subject))
    return(out)


def set_images(source_dir, target_dir, ratios, input_str='acq', label_str='brainmask' ):
    ''' This function takes a source directory, where the data is, and a
        ratio list (split test/train).
        It returns a pd.DataFrame that links file names to concepts, like
        t1 of subject 2 or pet-FDG for subject 15.
        This dataframe is exported is csv.

        Args:
            source_dir (str): the directory where the data is

        Returns:
            out (pd.DataFrame): a dataframe that synthesises the information
                of the source_dir.

    '''

    # 1 - gathering information (parsing the source directory)
    subject_dirs, pet_list, t1_list, names = gather_dirs(source_dir, input_str )
    # 2 - checking for potential errors
    if len(names) == 0:
        print_error_nosubject(source_dir)
    if sum(ratios) != 1:
        print_error_nosumone(ratios)

    # 3 - creating an empty directory of dataframes, then filling it.
    dfd = {}
    for name in names:
        data_subject = process(name, source_dir, pet_list, t1_list, label_str)
        dfd[name] = pd.DataFrame(data_subject)  # formerly subject_df
    # 4 - concatenation of the dict of df to a single df
    out = create_out(dfd)

    # 5 - attributing a test/train category for all subject
    out = attribute_category(out, ratios)

    # 6 - export and return
    if not exists(target_dir+os.sep+'report'): makedirs(target_dir+os.sep+'report') 
    out.to_csv(target_dir+os.sep+'report'+os.sep+'images.csv', index=False)
    return out


if __name__ == '__main__':
    set_images(argv[1], [0.7, 0.3])
