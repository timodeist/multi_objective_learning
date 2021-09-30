# Multi-Objective Learning to Predict Pareto Fronts Using Hypervolume Maximization

This repository contains the source code to train multiple neural networks for simple multi-objective (MO) regression as an illustration of the HV maximization approach described in arXiv preprint **[Multi-Objective Learning to Predict Pareto Fronts Using Hypervolume Maximization](https://arxiv.org/abs/2102.04523)**.

A full version of the manuscript's source code will be made available upon peer-review and publication of the manuscript.

Note: the source code is developed and tested on Linux platforms.

## Installation of dependencies
Install dependencies using the following command:


pip3 install --user -r requirements.txt

## MO regression example with 2 losses
The following script runs MO regression as explained in the paper using the HV maximization approach for 2 MSE losses and saves the output figures in the folder "output_files/mo_regression".


mo_regression_2obj.py

## MO regression example with 3 losses
The following script runs MO regression as explained in the paper using the HV maximization approach for 3 MSE losses and saves the output figures in the folder "output_files/mo_regression".


mo_regression_3obj.py

## Neural style transfer
Three content images without a source link in Table C2 (Deer, Dolomites, Sitojaure) are available in "style_transfer/content_images".

The generated images for all 25 image sets B1-B25 (see Table C2) are available in "style_transfer/generated_images_2d".
