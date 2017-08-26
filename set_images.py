import numpy as np
import pandas as pd
from re import sub
from sys import argv
from glob import glob
import os
from os.path import basename
from glob import glob
def set_images(source_dir,ratios):
    '''Creates a DataFrame that contains a list of all the subjects along with their PET images, T1 MRI images and labeled images.'''

    subject_dirs = glob(source_dir+os.sep+'*')
    pet_list = glob( source_dir + os.sep + '*' + os.sep + '*_acq*'  )
    t1_list = glob( source_dir + os.sep + '*' + os.sep + '*_t1.*'  )
    label_list = glob( source_dir + os.sep + '*' + os.sep + '*_labels_brainmask.*'  )
    names = [ basename(f) for f in  subject_dirs ]
    colnames=["subject",  "radiotracer", "pet", "t1", "label"]
    colnames=["subject",  "radiotracer", "task", "pet", "t1", "label"]
    nSubjects=len(names)
    out=pd.DataFrame(columns=colnames)
    for name in names:
        pet =  [ f for f in pet_list if name in f ]
        t1 =  [ f for f in t1_list if name in f ][0]
        pet_names = [ sub('.mnc', '', sub('acq-','', g)) for f in pet for g in f.split('_') if 'acq' in g ]
        task_names = [ sub('task-','', g) for f in pet for g in f.split('_') if 'task' in g ]
        labels=[]
        for p,t in zip(pet, task_names):
            label_fn = glob( source_dir + os.sep + '*' + os.sep + name +'*'+ t+ '*_labels_brainmask.*' )[0]
            labels.append(label_fn ) 
        n=len(pet)
        subject_df = pd.DataFrame(np.array([[name] * n,  pet_names, task_names , pet,[t1]*n, labels]).T, columns=colnames)
        out = pd.concat([out, subject_df ])

    nfolds=np.random.multinomial(nSubjects,ratios) #number of test/train subjects
    image_set = ['train'] * nfolds[0] + ['test'] * nfolds[1]
    out["category"] = "unknown"
    out.reset_index(inplace=True)
    l = out.groupby(["subject"]).category.count().values
    # please change later
    el = []
    for i in range(nSubjects):
       el.append([image_set[i]]*l[i])
    el = [l for sublist in el for l in sublist]
    out.category = el
    out.to_csv('images.csv')
    return out


if __name__ == '__main__':
    set_images(argv[1],[0.7,0.3])
