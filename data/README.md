# Datasets

To train our model we have used a private dataset. After fine-tuning SAM with it, we have tested our model on three hold-out datasets: OrgaQuant, OrganoSeg and OrganoID. 

## Private dataset

This private dataset contains images from different lens microscopes without ground-truth masks for the organoids. We have defined a fast semi-automatic tool to annotate the images. 

These images were mostly of size 2592 × 1944 pixels. We have used [Grounding DINO](https://arxiv.org/abs/2303.05499) and SAM ViT-Large to generate masks for them. After getting ground-truth framing boxes and masks we have created 48 patches per image with a sliding window without overlapping. The generated images have size 324x324 and the dimensions of the boxes and masks have been adjusted accordingly.

The dataset has been splitted in a train, validation and test split. For every sequence of images from a same record they are all on a same and unique split. Therefore, the validation and test split is non-seen data and they can be though as hold-out sets to evaluate the training performance.

The structure of the dataset looks like this:

```
    ·private
    |
    |-images
        |-...
    |-masks
    |   |-...
    |-metadata.json
```

 The `metadata.json` file contains all the information for every instance of the dataset: framing boxes, path of image and corresponding mask and dataset split.


### Private dataset generation

This semi-automatic tool to annotate the images from the private dataset can be seen in the notebook `notebooks/dataset_generation/private_dataset_generation.ipynb`. Follow the steps there to annotate a new image. 

## Orgaquant

[OrgaQuant](https://www.nature.com/articles/s41598-019-48874-y) dataset  contains human intestinal organoids images. It also contains information about the framing boxes for all the organoids in these images. We have generated the masks for the organoids with SAM ViT-Large given every original framing box as a prompt. 

This dataset is already given with a train and test split. We have used the test split as a hold-out dataset to test hour model.

The structure of the dataset looks like this:

```
    ·intestinal_organoid_dataset
    |
    |-test
    |   |-images
    |   |   |-...
    |   |-masks
    |   |   |-...
    |-train
    |   |-images
    |   |   |-...
    |   |-masks
    |   |   |-...
    |-metadata.json
    |-test_labels.csv
    |-train_labels.csv
```

 The files `train_labels.csv` and `test_labels.csv` contain the original framing boxes for every organoid in each image. The `metadata.json` file contains all the information for every instance of the dataset: framing boxes, path of image and corresponding mask and dataset split.


## Organoseg

## OrganoID