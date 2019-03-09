# AGNs

## Description

This repo contains an implementation of the *Adversarial Generative 
Nets (AGNs)*, that we proposed in our ACM TOPS 2019 paper (see
reference below). A demo for launching impersonation and dodging 
attacks against the VGG and OpenFace face recognition neural networks 
is provided.

## Data and models

Before you can run the code, you need to download the set of eyeglasses
that we synthesized from real textures (link: <https://goo.gl/Kr1HdV>; 
place them under `data/eyeglasses`) and the neural networks that we 
trained (link: <https://goo.gl/domYfN>; place them under `models/`).

## Dependencies

The code is implemented in MATLAB (we used `MATLAB R2015a`). As mentioned in the paper, our implementation depends on MatConvNet 
(<http://www.vlfeat.org/matconvnet/>)---a MATLAB toolbox for 
convolution neural networks. An extended version (containing additional 
layers, etc.) is provided under `dependencies/`.

To align images (necessary when running experiments with the OpenFace 
neural networks and for using new images in attacks), our code depends 
on Python packages (we used `python3.6`) for face and landmark 
detection. Specifically, the face and landmark detectors of the 
`dlib` package are used.

## Instructions for running

The attack code (see under `code/agn*.m`) takes face images that are 
aligned to VGG's canonical pose. For physical attacks, the aligned 
images should also contain green marks that are used for aligning the 
eyeglasses to the face image (see `data/demo-data2/` for examples).
See `demo.m` for examples of running attacks. 

For the purpose of the demo, aligned images are provided with the code.
To align new images, you can use the face- and landmark- detection, and face-alignment code under `dependencies/image-registration/`. Before
running the code, you need to update the paths in the files
`face_landmark_detection.m` and `openface_align.m`.  See `align_demo.m` 
for a face-alignment example.

## Reference

If you use the code, please cite our paper:

```
@article{Sharif19AGNs,
  author =       {Mahmood Sharif and Sruti Bhagavatula and 
  					  Lujo Bauer and Michael K. Reiter},
  title =        {A general framework for adversarial examples 
  					  with objectives},
  journal =      {ACM Transactions on Privacy and Security},
  year =         2019
}
```