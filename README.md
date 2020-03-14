In the repository , you will find code for encoding visual data (images) for HTM using
Keras auto-encoder.

I an using the spatial pooler to compare images in a unsupervised way.

To use the encoder ou will need a linux system (Ubuntu 18.04 x64) is preferred.
Start by downloading Mini-conda 3.7 x64, install conda tensorflow, pip install keras 
, pillow, numba, scikit-learn , numpy etc.
and finally head over to HTM CORE and install their implementation of HTM.

The repository and all the code is self-contained with no outside dependency for data, to 
run the example make sure you clone this repo to your home directory and maintain
current folder structure.

In the folder test_results, you will find images for number 5, I encode (5) image
and then train the spatial pooler across all encoded images. To test weather SP has 
been able to capture image data properly I provide SP with a sample (5) image and
ask SP to capture all images which look like number 5, .i.e. SP overlap function.

As proof,if the encoding mechanism is working properly then SP should return all
images which look like number 5. I have included examples for other number 4,8 etc
play around with those examples and tune the hyper parameters of RSDE and SP for
best results.  


I have provided comment in the code it self, for any other you can get in touch on
https://discourse.numenta.org/ my user id is aamir121a.

In the near future I will work on bigger auto encoder to encode complex RGB images
and keep this repository update with more examples.

I would very much welcome community ideas or code on improvement.