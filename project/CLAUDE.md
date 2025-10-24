# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a project on implementing a research project.

In order to understand the context in which we're working, please read the mission.md file.

If one were to simply train a VAE on the dataset, one would run into the problem that since most of the pixels are black background pixels, a simple solution that achieves good loss values (and high accuracy, 93%) is to simply predict black everywhere. To get around this problem, we implemented two additions to the loss:

    1. Focal loss, to focus on hard examples, which in our case are pixels with non-black colors.
    2. Class weighting, to address the class imbalance problem (we have way more black pixels than non-black pixels)

So whenever we see an accuracy of 93% we need to be suspicious because this probably just means that the model has collapsed again.

We have now established a beta schedule that, unlike earlier attempts, does not immediately collapse to black images, so in principle the approach might work.
However, we want to first make sure that our model can in principle represent the dataset (without beta). So next, we will create a slightly simpler version of the dataset, without the blob objects (i.e. only a handful of parameterized shapes of 10 different colors). 

For running code in the context of this project, use the zeus environment from pyenv like so:

pyenv activate zeus && python file.py

Whenever you write tests, put them in the appropriate location in the tests directory.