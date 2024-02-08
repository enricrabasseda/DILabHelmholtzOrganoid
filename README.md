# TUM-DI-LAB-Helmholtz

In this project we have fine-tuned SAM on Organoid data with novel topological data analysis techniques. We also present a semi-automatic tool for organoid image annotation.

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

### Semi-automatic organoid images annotation
The private organoid dataset that was used for the fine-tuning of SAM was not annotated. We have implemented a semi-automatic process for data annotation that can be replicated with the notebook `/notebooks/dataset_generation/private_dataset_generation.ipynb`.

For a more simplified version of the pipeline of Grounding DINO and SAM take a look at the notebook `notebooks/inference/dino_and_sam_inference.ipynb`. 

## Model training

In case of training the model it is possible to do it using a Topological loss and Geometrical loss. For more information regarding topological loss see the [original literature](https://arxiv.org/abs/2203.01703). To train the model using topological loss, adjust the hyperparameters and dataset location and run:

```
    python /utils/train_model/train_topo+geom_private.py
```

To train the model using only a geometrical loss, adjust the configuration and dataset location and run:

```
    python /utils/train_model/train_geom_private.py
```

These trained models will be saved in the folder `/models/`.

## Model evaluation

This method requires at least one model to compare in the folder `/models/`. In case of not having fine-tuned any SAM version, like specified above, download our fine-tuned version and save it in the mentioned folder. Depending on the hold-out dataset for which the model needs to be tested there are different python scripts. For example, for OrgaQuant dataset run:

```
    python /utils/evaluate_model/metrics_calculation_orgaquant.py
```

This will evaluate a fine-tuned model from `/models/` folder and compare it to MedSAM, SAM ViT-Base, SAM ViT-Large and SAM ViT-Huge computing different metrics. It is possible to adjust the evaluation preferences in the script. The results will be saved in the folder `/outputs/`.

## Inference

To test a fine-tuned model on organoid images given a manual box prompt you can use the notebook `notebooks/inference/model_inference.ipynb`. Please save the images on the right folder `/datasets/` and provide a box prompt for a correct usage of the models. 

In this last notebook it is also possible to compare the results of organoid detection for the fine-tuned model, MedSAM and SAM ViT-Base.

