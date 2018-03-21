# minc_keras
## About
minc_keras is a code base that was developped during a hackathon to facillitate the implementation of deep learning models for brain imaging with the Keras package.

#### Google Colab 

Create / Log-in to Google account 
Go to https://colab.research.google.com 
Download and load: https://tinyurl.com/yd8dd5x3

## Presentations

![NeurotechMTL -- Deep Learning with MRI (3.21.18)](./presentations/neurotechmtl_3.21.18_deeplearningwithmri.pdf) 


## Installation

### Docker (very easy):

Install docker on your OS: https://docs.docker.com/install/#cloud

docker pull tffunck/neurotech:latest

### DIY (pretty easy):
wget https://bootstrap.pypa.io/get-pip.py (Or go to the link and download manually)

python3 get-pip.py

pip3 install   pandas numpy scipy h5py matplotlib tensorflow keras

git clone https://github.com/tfunck/minc_keras

## Data

Data should be organized in the BIDS format (http://bids.neuroimaging.io/). While the code in this repository is in theory supports HDF5 files, at the moment only the MINC format is supported. Nifti support will be provided in future releases. 

#### Example Data :

data/output/

data/output/sub-01/sub-01_task-01_ses-01_T1w_anat_rsl.mnc

data/output/sub-01/sub-01_task-01_ses-01_variant-seg_rsl.mnc

data/output/sub-02/sub-02_task-01_ses-01_T1w_anat_rsl.mnc

data/output/sub-02/sub-02_task-01_ses-01_variant-seg_rsl.mnc


## Useage

#### Basic Useage:

python3 minc_keras/minc_keras.py --source /path/to/your/data/ --target /path/to/desired/output --epochs <number of epochs>  --input-str "string that identifies input files" --label-str "string that identifies labeled files" --predict <list of which subjects in test set> 

##### Example:
python3 minc_keras/minc_keras.py --source minc_keras/data/output/ --target . --epochs 5 --input-str "T1w_anat" --label-str "seg" --predict 1 



## Authors
Thomas Funck (thomas.funck@mail.mcgill.ca)

Paul Lemaitre

Andrew Doyle

