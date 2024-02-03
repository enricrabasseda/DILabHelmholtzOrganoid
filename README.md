# TUM-DI-LAB-Helmholtz

In this project we have fine-tuned SAM on Organoid data with novel topological data analysis techniques. 

## Introduction

Segment anything model (SAM) is currently the state-of the-art foundation model for image segmentation. It was trained on a set of 1 billion masks and aims to allow instance segmentation for any given image. 

SAM consists of an image encoder, a prompt encoder and a mask decoder. Hence SAM takes an image and a prompt as input and outputs a segmentation mask. This prompt can be a bounding box or a point in the image.

![SAM Architecture](https://github.com/facebookresearch/segment-anything/raw/main/assets/model_diagram.png?raw=true)

Recent papers started to fine-tune SAM on specific domains. In the medical domain Ma et al. and Zhang et al. presented fine-tuned SAM models: [MedSAM](https://arxiv.org/abs/2304.12306) and [SAMed](https://arxiv.org/abs/2304.13785). Ma et al. proposes to re-train the mask decoder, since it is commparatively small and therefore allows fast training. Zhang et al. proposes to use low-rank-based finetuning techniques to the image encoder. 

Organoids are miniature three-dimensional cell cultures that mimic real organs. Detecting organoids in images is crucial for advancing research, drug discovery, and disease modeling. It allows researchers to monitor their growth, structure, and response to treatments, aiding in the development of personalized medicine.

## Installation

### Install Grounding DINO

To use the code, please install **Grounding DINO** as it is indicated on their [repo](https://github.com/IDEA-Research/GroundingDINO/tree/main).

### Install required libraries

We recommend to set up a virtual environment and then install all the required packages from `requirements.txt`.

## Datasets

The model has been fine-tuned with a private dataset that cannot be provided. However, you can download the hold-out datasets to and place them in `/data/`. More information regarding the datasets structure can be found `/data/README.md`.

## Model training

In case of training the model it is possible to do it using a Topological loss and Geometrical loss. For more information regarding topological loss see the original literature. After adjusting the training preferences and the hyperparameters run:

```
    python /utils/run_train.py
```

## Model evaluation

This method requires at least one model to compare in the folder `/models/`. In case of not having fine-tuned any SAM version, like specified above, download our fine-tuned version and save it in the mentioned folder.Run:

```
    python /utils/evaluate_model.py
```

This will evaluate a fine-tuned model from `/models/`folder and compare it to MedSAM, SAM ViT-Base, SAM ViT-Large and SAM ViT-Huge computing different metrics. It is possible to adjust the evaluation preferences in the script.

## Inference

To run some inference cases with the fine-tuned models you can use the notebook `/notebooks/inference_demo.ipynb`. Please save the images on the right folder `/datasets/` and provide a box prompt for a correct usage of the models. 

In this notebook it is also possible to compare the results of organoid detection for a fine-tuned model, MedSAM and SAM ViT-Large.∫

