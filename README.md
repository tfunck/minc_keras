## Installation

### Docker (very easy):

Install docker on your OS: https://docs.docker.com/install/#cloud

docker pull tffunck/neurotech:latest

### DIY (pretty easy):
wget https://bootstrap.pypa.io/get-pip.py (Or go to the link and download manually)

python3 get-pip.py

pip3 install   pandas numpy scipy h5py matplotlib tensorflow keras

git clone https://github.com/tfunck/minc_keras

### Data

Data should be organized in the BIDS format (http://bids.neuroimaging.io/). While the code in this repository is in theory supports HDF5 files, at the moment only the MINC format is supported. Nifti support will be provided in future releases. 

#### Example :
data/
data/sub-01/sub-01_task-01_ses-01_T1w.mnc
data/sub-01/sub-01_task-01_ses-01_labels.mnc
data/sub-02/sub-02_task-01_ses-01_T1w.mnc
data/sub-02/sub-02_task-01_ses-01_labels.mnc

#### Example Data


### Useage

#### Basic Useage:

python3 minc_keras/minc_keras.py --source /path/to/your/data/ --target /path/to/desired/output --epochs <number of epochs>  --input-str "string that identifies input files" --label-str "string that identifies labeled files" --predict <list of which subjects in test set> 

##### Example:
python3 minc_keras/minc_keras.py --source data/ --target /hcp/results/ --epochs 5 --input-str "T1w" --label-str "label" --predict 1 


