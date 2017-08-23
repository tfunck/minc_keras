# Deep Pet -- Deep Brain Hackathon 2017

## Background
Brain masks are important for PET image processing.

PET brain masks can help improve co-registration to T1 images

## Objective
Create brain masks for PET images across a wide variety of PET images.

## Data Set
### Open
1) Simulated PET images: 15 subjects x 3 radiotracers (FDG, L-DOPA, Raclopride)

<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/fdg.png" width=100 "FDG PET">
<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/fdopa.png" width=100 "FDOPA PET">
<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/raclopride.png" width=100 "Raclopride PET">
<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/t1.png" width=100 "T1">
<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/brainmask.png" width=100 "Brain Mask">

2) ADNI - ???
### Closed
3) 46 subjects x FMZ , 31 x FDG, 26 x Raclopride

Data is already formatted and ready to go. Basic python program with Keras for implementing CNN.


## Goals

1) Test different network architectures
2) Test different features (e.g., full 3D volumes, 2D slices, 1D profiles)







pet_brainmask_convnet

