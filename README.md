# SRCNN-keras
This repository is implementation of the "Image Super-Resolution Using Deep Convolutional Networks".

My implementation have some difference with the original paper, include:

use SGD for optimization, with learning rate 0.0003 for all layers.
Use the opencv library to produce the training data and test data, not the matlab library. This difference may caused some deteriorate on the final results.
I trained and tested with my own single channel images.

So if you want to compare your results with some academic paper, you may want to use the code written with matlab.


training and test:
Please prepare your own data, cut them into 32 * 32 size, and convert them into .npy format. In fact, other sizes are OK.

Net
We used the network settings for experiments, i.e., f1=9,f2=5,f3=5,n1=128,n2=64,n3=1.

Just run  main.py
