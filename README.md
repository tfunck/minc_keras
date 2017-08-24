# Deep Pet -- Deep Brain Hackathon 2017

## Background
PET is an imaging modality that uses a radioactive tracer attached to a biological metabolite to quantify the structure and function of the brain. Brain masks are important for PET image processing. They can, for example, in helping improve PET-MRI coregistration. However, it is dificult produce a "one size fits all" algorithm for deriving brain masks from PET images that will apply across a wide variety of different radiotracers. This is because the signal distribution varies greatly from one tracer to the next. Deep neural networks may have sufficient flexibility to allow for a generic brain extraction algorithm for PET.

## Objective
Create brain masks for PET images that performs accureately across a wide variety of types of PET images.

## Data Set
The data for the project comes from three sources, two of which are open-access.

### Open
1) Simulated PET images: 15 subjects x 3 radiotracers (FDG, L-DOPA, Raclopride)

<img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/readme/fdg.png" alt="FDG PET" width=150 > <img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/readme/fdopa.png" width=150 > <img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/readme/raclopride.png" width=150> <img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/readme/t1.png" width=150 > <img src="https://github.com/tfunck/pet_brainmask_convnet/blob/master/readme/brainmask.png" width=150>

2) ADNI (Not sure yet)
### Closed
3) 46 subjects x FMZ , 31 x FDG, 26 x Raclopride

Data is already formatted and ready to go. Basic python program with Keras for implementing CNN.


## Goals

1) Find optimal network architecture
2) Evaluate different features (e.g., full 3D volumes, 2D slices, 1D profiles)
3) Determine if network generalizes to radiotracers on which it has not been trained
4) Determine if training on MRI as well as PET helps improve performance



