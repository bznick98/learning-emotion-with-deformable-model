![](https://visitor-badge.glitch.me/badge?page_id=bznick98.learning-emotion-with-deformable-model)
# Learning Face Emotions with Deformable Model

The repo for 2021 Fall CS269 Final Project: Learning Face Emotions with Deformable Model. 
- Group Member:
Yu Hou, Xueer Li, Zongnan Bao, Xiaoyang Yu, Felix Zhang

# Dataset
- [**FER2013**](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
  - 48x48
  - *large*
  - 28709 training images
  - 3589 validation images
  - 7 labels
- [**FER_CK_PLUS**](https://www.kaggle.com/sudarshanvaidya/corrective-reannotation-of-fer-ck-kdef)
  - 48x48
  - *large*
  - 8 labels, ignoring `contempt` since it's not in FER2013
  - The rest: `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutrality`
  - cleaned version of FER2013, plus CK+ images and a dataset called kdef?
  - higher quality than FER2013
- [**CK_PLUS**](https://drive.google.com/drive/folders/1W-dl_w1ynzEDUhiOCMjbCcYwmaoZppRN?usp=sharing)
  - 256x256
  - *small*
  - 297 training images
  - 32 validation images
  - 7 labels: `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutrality`
  - unstable validation metrics since validations set is fairly small :(
  - achieves highest accuracy (easist to train?)
  - NOTE: be sure to add `-resize 224` if you want to use CK_PLUS(256x256) on models that expect 224 input, another option is to add `-rcrop_size 224`

# Models
### Architectures
- **Deep_Emotion**
  - have both 224x224-input version and 48x48-input version.
  - if choosing 224x224 as input, large portion of parameters will be in fc layers. 
  - added BatchNorm after every Conv/FC compared to original deep emotion.
- **Deep_Emotion224**
  - dedicated for 224x224 input.
  - added/re-ordered layers, so that parameters are more evenly spread between Conv and FC layer.
  - No significant performance boost.
- **VGG**
  - really big model, might overfit.
  - need to experiment more on this type of network
- **Simple_CNN**
  - 3 Conv layers + 2 FC layers
  - simple network to test avoid overfitting

### Optional Components *(recommended)*
- Deformable Convolution Layers
  - Can be enabled by setting `-dc` or `--de_conv`
  - Improves `val acc` by roughly 3%~4%.
  - Theoretically improves localization ability of the model.
- Wider Deep Emotion
  - Can be enabled by setting `-wide` or `--wide`
  - Original deep emotion has `channel=10` for Conv layers, enabling this changes `channel=64`, have bigger model capacity.
- Dropout Layer
  - By default there's 1 dropout layer after 1st fc layer.
  - Can add 1 extra by setting `-n_drop 2`, will be after 2nd(final) fc layer
  - All dropout rates are controlled by `-drop X`, X in [0, 1], 0=`no dropout`

# Training
### Training tricks
- Learning Rate Scheduler
  - Can be enabled by setting `-lrsc`
  - Uses PyTorch's `ReduceLROnPlateau()`
  - With `patience=20` and `min_lr=1e-7`
- Weight Decay
  - Can be enabled by setting `-wd X` or `--weight_decay X`
  - X default = `1e-4`
  - L2 Regularization, reduce overfitting.
- Data Augmentations
  - `RandomHorizontalFlip(0.5)`
    - Can be enabled by setting `-hflip`
  - `RandomCrop(X)`
    - Can be enabled by setting `-rcrop`, X can be set by `-rcrop_size X`
    - X => int, Cropped to size (X,X)
  - `RandomColorJitter(X,Y,Z)`
    - Can be enabled by setting `-rjitter`
    - X in [0,1], jittering brightness, can be set by `-rjitter_b X`
- Data Processing
  - `Resize`
    - Can be set by `-resize X`
    - Both training and validation images will be resized to (X, X)

# How to Run
### Train
- `-d` specifies the directory stored specific dataset, like `fer2013/`,  `fer_ckplus_kef/` and `CK_PLUS/`. I put all those data under `./data/`, so `-d data/CK_PLUS` if we want to use `CK_PLUS`.
- `-ds`, acutually a little bit redundant? specifies which dataset we want to use, a `str`, can only be one of {`FER2013`, `FER_CKPLUS`, `CK_PLUS`}
- `-m` specifies model we want to use, can only be one of {`de`, `de224`, `vgg`, `simple`}
- Example: if want to run on dataset `CK_PLUS` using `Deep Emotion` with some tricks added, we can: 

```python train.py -d data/CK_PLUS -ds CK_PLUS -m de -dc -wide -n_drop 2 -drop 0.4 -rcrop -rcrop_size 224 -hflip -resize 48```

### Run on colab (CLI version)
* https://colab.research.google.com/drive/18ZvSw12sYvG73KlNJ2n77IGPkJln1HO7?usp=sharing

### Project Structures
* all DL model classes are under models/
* all custom dataset classes are under datasets/
* data can be put into data/
* some utils used for training are put under utils/


# Experiments
Link to experiments I've done: https://ppnk.notion.site/CS269-Final-Project-Experiments-1e0e15bde4134825b58f0ec8257bd1bd

# TODOs?
- [ ] Find out a suitbale/best performance network.
- [ ] Investigate more on STN part in deep emotion.
- [ ] Combine Deformable Model (contours/landmark) to boost DL model performance.
- [ ] Data Augmentation part of **parser** are a little redundant, especially `-rcrop` and `-rcrop_size`, only need one.
