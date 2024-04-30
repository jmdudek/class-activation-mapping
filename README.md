# Class Activation Mapping 

This repository contains the code and presentation for our ([@GerritBartels](https://github.com/GerritBartels) and [@jmdudek](https://github.com/jmdudek)) topic in Deep Neural Network Analysis. We have implemented Class Activation Mapping (CAM), Grad-CAM, Grad-CAM++, and their guided versions. These techniques are applied to various Convolutional Neural Networks (CNNs) trained on the ImageNet dataset.

## Table of Contents
- [Repository Usage](#repository-usage)
- [Overview](#overview)
- [Examples](#examples)
    - [CAM](#cam)
    - [Counterfactual Explanation using Grad-CAM](#counterfactual-explanation-using-grad-cam)
    - [Guided Grad-CAM](#guided-grad-cam)
- [References](#references)

## Repository Usage

1. Clone the repository:
    ```shell
    $ git clone https://github.com/jmdudek/class-activation-mapping
    ```

2. Navigate to the project directory:
    ```shell
    $ cd class-activation-mapping
    ```

3. Set up a virtual environment with Python 3.11.* and install the required packages from the `requirements.txt` file:
    ```shell
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```
    or with conda:
    ```shell
    $ conda create --name cam python=3.11
    $ conda activate cam
    $ pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook `cam.ipynb` to try out the CAM, Grad-CAM and Grad-CAM++ visualizations, as well as the guided variants for the latter two.

## Overview

Class Activation Mapping (CAM) is a technique used to visualize the regions of an input image that contribute the most to the prediction made by a CNN. Grad-CAM and Grad-CAM++ are extensions of the original CAM mathod that are more versatile and capable. The guided versions of Grad-CAM and Grad-CAM++ further enhance the interpretability of the visualizations by combining the class descriptive power of both methods with the high resolution from Guided Backpropagation.

An overview over all three CAM methods can be seen in the following image:

![CAM Visualization](/images/cam_gc_gc++.png)

## Examples

### CAM

![CAM Visualization](/images/cam_elephant.png)


### Counterfactual Explanation using Grad-CAM

![Counterfactual Explanation](/images/counterfactual_explanation.png)


### Guided Grad-CAM

![Guided Grad-CAM](/images/guided_gc.png)


## References
[1] Zhou, Bolei, et al. "Learning deep features for discriminative localization." *Proceedings of the IEEE conference on computer vision and pattern recognition.* 2016. https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

[2] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." *Proceedings of the IEEE international conference on computer vision.* 2017. https://arxiv.org/pdf/1610.02391.pdf

[3]  Chattopadhyay, Aditya, et al. "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks." *Proceedings of the IEEE Winter Conf. on Applications of Computer Vision.* 2019. https://arxiv.org/pdf/1710.11063.pdf

[4] All other images are from: https://unsplash.com/
