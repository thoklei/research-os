# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a project on implementing a research project.

In order to understand the context in which we're working, please read the mission.md file.

If one were to simply train a VAE on the dataset, one would run into the problem that since most of the pixels are black background pixels, a simple solution that achieves good loss values (and high accuracy, 93%) is to simply predict black everywhere. To get around this problem, we implemented two additions to the loss:

    1. Focal loss, to focus on hard examples, which in our case are pixels with non-black colors.
    2. Class weighting, to address the class imbalance problem (we have way more black pixels than non-black pixels)

So whenever we see an accuracy of 93% we need to be suspicious because this probably just means that the model has collapsed again.
 
We are currently stuck at the following problem: We want to validate our model's capacity by first overfitting a single batch. However, this part is not working properly, due to some weird mode collapse issue: When overfitting a single batch, the reconstructed images are just black, because the model learns the suboptimal solution of just predicting the most likely pixel value. We encountered this problem before, in the normal training, where we got around it by adding a class weighting term to the loss. In principle this should work for the single batch we're overfitting as well, but in practice it does not. 


For running code in the context of this project, use the zeus environment from pyenv like so:

pyenv activate zeus && python file.py