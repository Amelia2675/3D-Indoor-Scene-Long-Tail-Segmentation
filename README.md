# 3D-Indoor-Scene-Long-Tail-Segmentation
Language-grounded Indoor 3D semantic segmentation in the wild


## 1. Language ground 3D pretraining
Pretrain the 3D feature extractor by  contrastive loss between text anchors and 3D features. (set lambda = 1)

Training objective:

<img width="387" alt="image" src="https://user-images.githubusercontent.com/109503040/225255443-6eadcdc3-3b0b-475e-929b-a2d8b894069d.png">

## 2. 3D Semantic Segmentation Fine-tuning
Finetune the 3D semantic segmentation model with category-balanced focal loss and instance sampling.

### Category-balanced focal loss
Add a modulating factor for a cross entropy loss

<img width="338" alt="image" src="https://user-images.githubusercontent.com/109503040/225255657-26700bc4-e849-4884-a141-f9026819373b.png">

where pt is the prediction probability for respective target label and gamma is a constant to control the factor (set gamma to 2) 

### Instance Sampling
Place instances from less-seen categories (tail)  and break overly specific dependencies for recognition. 
We tried instance sampling in fine-tuning stage but the result is much worse than another settings, so we decided to add it in stage 3 training to see if it would work.

<img width="525" alt="image" src="https://user-images.githubusercontent.com/109503040/225255826-bee6ff09-c954-4552-8414-110846a35eb4.png">

### Mixing technique
Create new training samples by combining two augmented scene. Object instances are implicitly placed into novel out-of-context environments. Thus, the semantics can be inferred from local structure.
We tried this method but didn't get improvement on the results.

## Experiments
We trained the model with different settings for 200 epochs each stage. (ce stands for cross-entropy loss and focal stands for mentioned category-balanced focal loss. While ts means adding instance sampling for tail categories in data augmentation.)

<img width="570" alt="image" src="https://user-images.githubusercontent.com/109503040/225256142-d1a790f4-5bc1-404b-b6a3-605a4c1865f8.png">

## Results
1.Focal loss can improve the IoU of tail and common categories significantly, comparing to cross-entropy loss.

2.Pretrain the 3D feature extractor yields better result.

3.Tail sampling cannot improve the performance according to our experiments.

# How to run this code?

## Prerequisite

```sh
cd LanguageGroundedSemseg/
# create conda environment and activate
conda env create -f config/lg_semseg.yml
conda activate lg_semseg
# download dataset and checkpoint
source download.sh
```
Since we have self-generated extra dataset, we hardcode dataset path in .sh files and you can download it from ``download.sh``. To use your own dataset path for inferencing please modify dataset path in ``inference.sh``. For training, also make sure to turn off sample_tail_instance in ``.sh`` files if you use your own dataset.

Additionally, [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) has to be installed manually with a specified CUDA version. 
E.g. for CUDA 11.1 run 

```sh
export CUDA_HOME=/usr/local/cuda-11.1 # your cuda version
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas=openblas"
# if encounter
# AttributeError: module 'distutils' has no attribute 'version'
# reinstall setuptools
pip uninstall setuptools
pip install setuptools==59.5.0
```

## Run Inference

```sh
source inference.sh <TEST_NAME_POSTFIX> <ADDITIONAL_ARGS>
# e.g. source inference.sh baseline-test
```

output data will be dumped to ``./output/Scannet200Voxelization2cmDataset/Res16UNet34D-<TEST_NAME_POSTFIX>/visualize/fulleval/`` in ``.ply`` and ``.txt`` format.

to remove ``.ply`` under ``fulleval/`` folder, you can simply run

```sh
python3 get_upload_txt.py <PATH_TO_FULLEVAL_FOLDER>
```

## Run Visualization

You can visualize plyfile after inferencing, just run

```sh
python3 visualize.py <plt_filepath>
# e.g. python3 visualize.py ./output/Scannet200Voxelization2cmDataset/Res16UNet34D-baseline-test/visualize/fulleval/scene0500_00.ply
```

## Run Training

### Language Grounded Pretraining (stage 1)

```sh
source training_scripts/text_representation_train.sh <BATCH_SIZE> <TRAIN_NAME_POSTFIX> <ADDITIONAL_ARGS>
# e.g. source training_scripts/text_representation_train.sh 2 baseline-stage_1
```

### Downstream Semantic Segmentation (stage 2)

For this stage modify environment variables ``PRETRAINED_WEIGHTS`` in ``training_scripts/train_models.sh``, then run

```sh
source training_scripts/train_models.sh <BATCH_SIZE> <LOSS_TYPE> <TRAIN_NAME_POSTFIX> <ADDITIONAL_ARGS>
# e.g. source training_scripts/train_models.sh 2 cross_entropy baseline-stage2
```

``<LOSS_TYPE>`` can be ``focal`` or ``cross_entropy``

### Finetune Stage

For this stage modify environment variables ``PRETRAINED_WEIGHTS`` in ``training_scripts/fine_tune_classifier.sh``, then run

```sh
source training_scripts/fine_tune_classifier.sh <BATCH_SIZE> <LOSS_TYPE> <SAMPLE_TAIL> <TRAIN_NAME_POSTFIX> <ADDITIONAL_ARGS>
# e.g. source training_scripts/fine_tune_classifier.sh 2 focal True baseline-finetune
```
``<SAMPLE_TAIL>`` can be ``True`` or ``False``
