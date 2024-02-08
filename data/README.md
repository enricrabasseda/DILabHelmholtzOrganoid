# Datasets

To train our model we have used a private dataset. After fine-tuning SAM with it, we have tested our model on three hold-out datasets: OrgaQuant, OrganoSeg and OrganoID. 

Here are the dimensions of the datasets:

| **Dataset** | **Train** | **Validation** | **Test** | **Total** |
|:-----------:|:---------:|:--------------:|:--------:|:---------:|
| Private     |   16134   |      4630      |   4399   | **25163** |
| OrgaQuant   |   13004   |        -       |   1135   | **14139** |
| OrganoSeg   |    999    |       350      |   374    | **1723**  |

## Private dataset

This private dataset contains images from different lens microscopes without ground-truth masks for the organoids. We have defined a fast semi-automatic tool to annotate the images. 

These images were mostly of size 2592 × 1944 pixels. We have used [Grounding DINO](https://arxiv.org/abs/2303.05499) and SAM ViT-Large to generate masks for them. After getting ground-truth framing boxes and masks we have created 48 patches per image with a sliding window without overlapping. The generated images have size 324x324 and the dimensions of the boxes and masks have been adjusted accordingly.

The dataset has been splitted in a train, validation and test split. For every sequence of images from a same record they are all on a same and unique split. Therefore, the validation and test split is non-seen data and they can be thought as hold-out sets to evaluate the training performance.

Unfortunately, this dataset cannot be provided.

### Private dataset generation

This semi-automatic tool to annotate the images from the private dataset can be seen in the notebook `notebooks/dataset_generation/private_dataset_generation.ipynb`. Follow the steps there to annotate a new image of organoids.

## Orgaquant

[OrgaQuant](https://www.nature.com/articles/s41598-019-48874-y) dataset  contains human intestinal organoids images. It also contains information about the framing boxes for all the organoids in these images. We have generated the masks for the organoids with SAM ViT-Large given every original framing box as a prompt. 

This dataset is already given with a train and test split. We have used the test split as a hold-out dataset to test our model.

The structure of the dataset looks like this:

```
    intestinal_organoid_dataset
    |---metadata.json
    |---test_labels.csv
    |---train_labels.csv
    |---test
    |   |---images
    |   |   |---...
    |   |---masks
    |       |---...
    |---train
        |---images
        |   |---...
        |---masks
            |---...

```

 The files `train_labels.csv` and `test_labels.csv` contain the original framing boxes for every organoid in each image. The `metadata.json` file contains all the information for every instance of the dataset: framing boxes, path of organoid's image, corresponding mask and dataset split.


## OrganoSeg

[OrganoSeg](https://www.nature.com/articles/s41598-017-18815-8) presented a dataset with colon and colorectal-cancer organoid morphologies. The original images had resolution 864x648 and also had a mask to identify ground-truth organoids in the image. We have created patches using a sliding window with overlap and for every image we have generated 4 patches of size 432x432. Same work has been done for the masks.

To get instance segmentation masks we have used Connected Component Analysis to connect one unique mask to every organoid. The code can be seen in `/notebooks/dataset_generation/organoseg_dataset_generation.ipynb`. We have filtered out of the dataset all masks that are too small to be detected or cannot be clearly recognized from the image. To do that we kept masks that have an area bigger than 2000 pixels.

The structure of the dataset looks like this:

```
    colon_dataset
    |---metadata_semantic_segmentation.json
    |---metadata_instance_segmentation.json
    |---original
    |   |---colon_images
    |   |   |---...
    |   |---colon_masks
    |       |---...
    |---augmented
        |---colon_images
        |   |---...
        |---colon_instance_masks
        |   |---...
        |---colon_masks
            |---...

```

 The `metadata_semantic_segmentation.json` contains all metadata to use this dataset for semantic segmentation objectives: paths of images and corresponding masks. The file `metadata_instance_segmentation.json` contain all the information for instance segmentation objectives: framing boxes, path of organoid's image, corresponding individual mask and dataset split.