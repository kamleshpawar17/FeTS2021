# Federated Tumor Segmentation Challenge 2021 (task-2 2nd place solution)

## Installation
clone the repository and run the setup.sh file
````
sh setup.sh
````

## Data Prepration for training only
set the ```src_path''' to the nifiti images and ```dst_path``` to ouput dir for .npy data
```
generate_numpy_data.py --src_path=./data/nifti/train/*/*seg.nii.gz --dst_path=./data/np/train/
```

## Training
````
cd ./trainingScripts
sh train.sh
````

## Inference
````
inference.py
````

## Paper:
Pawar, K., Zhong, S., Chen, Z., Egan, G. (2022). Brain Tumor Segmentation Using Two-Stage Convolutional Neural Network for Federated Evaluation. In: Crimi, A., Bakas, S. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2021. Lecture Notes in Computer Science, vol 12963. Springer, Cham. https://doi.org/10.1007/978-3-031-09002-8_43

https://link.springer.com/chapter/10.1007/978-3-031-09002-8_43

