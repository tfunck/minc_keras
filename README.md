# minc_keras
## About
minc_keras is a code base that was developped during a hackathon to facillitate the implementation of deep learning models for brain imaging with the Keras package. It is also used as a hands-on teaching tool for the presentations listed below. 

## MAIN 2018 :
### [Presentation](./presentations/MAIN2018_segmentation.pdf)
### [Colab](https://colab.research.google.com/github/tfunck/minc_keras/blob/master/main2018.ipynb)

### Collaborators
<img src=sponsors/neurotechx.png align="right" alt=neurotechx width=100>\
<img src=sponsors/mcin.png align="right" alt=mcin width=150>
Presentations were created in collaboration with the [MCIN](https://mcin-cnim.ca/) lab and [NeuroTechX](https://neurotechx.com/). NeuroTechX is a non-profit organization whose mission is to facilitate the advancement of neurotechnology by providing key resources and learning opportunities, and by being leaders in local and worldwide technological initiatives. Their 3 pillars are “Community”, “Education”, and “Professional Development”.



## Presentations

### Deep Learning with MRI
#### Version: 28.08.18
![Workshop 1 (Part 1 & 4) -- Deep Learning with MRI ](./presentations/neurotechmtl_28.8.18_deep_learning_with_mri.pdf) \
![Workshop 1 (Part 2) -- Intro to ML ](./presentations/neurotechmtl_28.8.18_suarez_intro_to_ml.pdf) \
![Workshop 1 (Part 3) -- Intro to Neural Networks ](./presentations/neurotechmtl_28.8.18_doyle_intro_to_neural_nets.pdf) 

#### Version: 21.03.18

![Workshop 1 -- Deep Learning with MRI (21.3.18)](./presentations/neurotechmtl_21.3.18_deep_learning_with_mri.pdf) \
![Workshop 1 -- Intro to ML (21.3.18)](./presentations/IntroML.pdf)

### More coming soon...

## Installation

### Google Colab (best)

Create / Log-in to Google account \
Go to https://colab.research.google.com \
Download and load: https://www.dropbox.com/s/8uw13lbwbf83c0d/NeuroTech_MTL_28_8_18.ipynb?dl=0

#### Docker (very easy):

Install docker on your OS: https://docs.docker.com/install/#cloud

docker pull tffunck/neurotech:latest

#### DIY (pretty easy):
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

## Support provided by

### [Montreal Neurological Institute](https://www.mcgill.ca/neuro/)

![MNI](sponsors/mni.jpg)

### [Ludmer Centre](http://ludmercentre.ca/)
![Ludmer](sponsors/ludmer.png)

### [MCIN](https://mcin-cnim.ca/)
![MCIN](sponsors/mcin.png)

### [NeuroTechX](https://neurotechx.com/)
![NeuroTechX](sponsors/neurotechx.png)

## Authors
Thomas Funck (thomas.funck@mail.mcgill.ca)

Paul Lemaitre

Andrew Doyle

