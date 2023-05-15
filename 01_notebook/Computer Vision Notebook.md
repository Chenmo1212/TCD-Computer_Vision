# Computer Vision Notebook

## Introduction

### What is computer vision?

Every image tells a story. But it is hard for a computer to understand what happened. 

- Try to perceive the "story" behind a picture is the goal of computer vision.
- Compute properties of the world  
  - 3D shape
  - Names of people or objects
  - what happened?

At present, Computer Vision can not match human perception. Human perception has its shortcoming, like human quite often are fooled by visual illusion. But humans can tell a lot about a scene from a little information.

### The goal of Computer Vision

- Compute the 3D shape of the world
- Recognize objects and people 
- “Enhance” images  
- Forensics  
- Improve photos (“Computational Photography”)  

### Why study Computer Vision?

- Billions of images/videos captured per day  
- Huge number of potential applications
  - Optical character recognition  
  - Face detection 
  - City-scale 3D reconstruction  
  - Face analysis and recognition  
  - Vision-based biometrics  
  - Login without a password  
  - Bird identification  
  - Special effects: shape capture  
  - Special effects: motion capture  
  - 3D face tracking w/ consumer cameras  
  - Image synthesis  
  - Sports
  - Smart Cars
  - self-driving cars
  - Robotics 
  - Medical imaging  
  - Virtual & Augmented Reality 

### Why is computer vision difficult?  

- Intra-class variation  
- Motion  
- Background clutter  
- Occlusion  
- local ambiguity  



## Image Filtering, Resampling and Interpoation  

### What is an image?  

- Record of light rays
- Grid of pixel values (digital camera)  
  - common to use one byte per value: 0 = black, 255 = white
- The eye

### Filters

> Filtering: Form a new image whose pixel values are a combination of the original pixel values.

- To get useful information from images 
  - E.g., extract edges or contours (to understand shape)  
- To enhance the image
  - E.g., to remove noise
  - E.g., to sharpen and “enhance image” a la CSI
- A key operator in Convolutional Neural Networks  

#### Image Processing problems

- Image Restoration
  - denoising
  - deblurring
- Image Compression
  - JPEG, HEIF, MPEG, …
- Computing Field Properties
  - optical flow
  - disparity
- Locating Structural Features
  - corners
  - edges  

#### Question: Noise reduction 

- Q: Given a camera and a still scene, how can you reduce noise?  
- A: Take lots of images and average them! 

#### Image filtering 

Modify the pixels in an image based on some function of a local neighborhood of each pixel.

One simple version of filtering: linear filtering (cross-correlation, convolution)

-  Replace each pixel by a linear combination (a weighted sum) of its neighbors  

> The prescription for the linear combination is called the“`kernel`” (or “mask”, “filter”)

Different `kernel`s:

- Mean filtering 
- Gaussian 
- 
