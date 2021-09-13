# Data Prepare

### Get facial landmarks

We use the dense landmarks extracted by [Face++ Dense Facial Landmarks](https://www.faceplusplus.com/sdk/densekey/) to align images and train our FlowNet. You can process your images to get the landmarks by: 

    # change the image path in get_landmarks.py and add the face++ keys
    python get_landmarks.py

To speed up the processing, we only extract the landmarks under `07` illumination condition from each pose. The naming format is `xxx_xx_xx_xxx.json` (no illumination).

### Get facial masks

We use [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) to extract the facial masks. The mask is composed of the segmentation of face, ear, hair, and neck. To speed up the processing, we only extract the masks under `07` illumination condition from each pose. The naming format is `xxx_xx_xx_xxx_07.png`.

**The landmarks and masks we used to process multipie dataset can download from [GoogleDrive](https://drive.google.com/drive/folders/1U26FvuLtXraxPrRNxCxFka3DOvRuG4NW?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1X62Atd9Q_USs0aGQk3WllA)(l98p).**

### Process

Removing the comments and change the path in `process.py`, then run

    python process.py
    
The final dataset folder is structured as follows:

    dataset
        ├── multipie
        │       ├── train
        │       │     ├── images
        │       │     ├── masks
        │       │     └── landmarks.npy
        │       └── test
        │             ├── images
        │             ├── gallery_list.npy (optional)
        │             └── visual_list.npy (optional)
        └── lfw
             ├── images
             └── pairs.txt