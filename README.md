# GCRCR: Gated Convolutional and Recurrent Character Recognition

## Overview

Handwritten Text Recognition (HTR) is a complex task due to the diversity of handwriting styles, stroke variations, slants, and spacing irregularities. Traditional methods struggle with these variations, especially when handling non-standardized or degraded inputs. This repository contains the implementation of the Gated Convolutional and Recurrent Character Recognition (GCRCR) model, which effectively addresses these challenges using an advanced preprocessing pipeline and a hybrid deep learning architecture.

## Features

- Preprocessing Pipeline: Includes padding, desloping, deslanting, and data augmentation to standardize input images.
- Model Architecture:
  - GCRCR: Combines gated convolutional neural networks with a bidirectional recurrent neural network (BiRNN) for robust feature extraction and sequence modeling.
  - FCCR: A Fully Convolutional Character Recognition model as a baseline for comparison.
- CTC Loss Function: Utilized to handle variable-length output sequences during training.
- Performance Metrics: The GCRCR model achieves 4.14% CER (Character Error Rate) and 8.03% WER (Word Error Rate) on the IAM Handwriting Database, outperforming existing methods by 3.7% (CER) and 4.2% (WER).

## Dataset

The IAM Handwriting Database is used for training and evaluation. The dataset can be accessed from:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
`
pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn jiwer autocorrect
`

## Results

The GCRCR model demonstrates superior performance in handwritten text recognition tasks:
- Character Error Rate (CER): 4.14%
- Word Error Rate (WER): 8.03%
