## GRIT: Faster and Better Image captioning Transformer (ECCV 2022)

This is the code implementation for the paper titled: "GRIT: Faster and Better Image-captioning Transformer Using Dual Visual Features" (Accepted to ECCV 2022) [[Arxiv](https://arxiv.org/abs/2207.09666)].


## Introduction

This paper proposes a Transformer neural architecture, dubbed <b>GRIT</b> (Grid- and Region-based Image captioning Transformer), that effectively utilizes the two visual features to generate better captions. GRIT replaces the CNN-based detector employed in previous methods with a DETR-based one, making it computationally faster.


<div align=center>  
<img src='.github/grit.png' width="100%">
</div>


## Model Zoo
| Model                                           | Task             | Checkpoint                                                                                           |
|-------------------------------------------------|------------------|------------------------------------------------------------------------------------------------------|
| Pretrained object detector (A) on Visual Genome | Object Detection | [GG Drive link](https://drive.google.com/file/d/1ZWPovkK5YhxtyCaVULCTNoPu8Jd-MKGh/view?usp=share_link)  |
| Pretrained object detector (B) on 4 OD datasets | Object Detection | [GG Drive link](https://drive.google.com/file/d/1xERJN3CvQcUcwgRZd31CUsnep_xnELcs/view?usp=share_link)  |
| GRIT (using the object detector A)              | Image Captioning | [GG Drive link](https://drive.google.com/file/d/12tsI3Meka2mNLON-tWTnVJnUzUOa-foW/view?usp=share_link) |
| GRIT (using the object detector B)              | Image Captioning | [GG Drive link](https://drive.google.com/file/d/1jgEqNFuKcKg_RcG4Nq8bhWvCgzi6bjuD/view?usp=share_link) |

## Installation

### Requirements
* Python >= 3.9, CUDA >= 11.3
* PyTorch >= 1.12.0, torchvision >= 0.6.1
* Other packages: pycocotools, tensorboard, tqdm, h5py, nltk, einops, hydra, spacy, and timm

* First, clone the repository locally:
```shell
git clone https://github.com/davidnvq/grit.git
cd grit
```
* Then, create an environment and install PyTorch and torchvision:
```shell
conda create -n grit python=3.9
conda activate grit
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# ^ if the CUDA version is not compatible with your system; visit pytorch.org for compatible matches.
```
* Install other requirements:
```shell
pip install -r requirements.txt
python -m spacy download en
```
* Install Deformable Attention:
```shell
cd models/ops/
python setup.py build develop
python test.py
```

## Usage


### Data preparation

Download and extract COCO 2014 for image captioning including train, val, and test images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco_caption/
├── annotations/  # annotation json files and Karapthy files
├── train2014/    # train images
├── val2014/      # val images
└── test2014/     # test images
```
* Copy the files in `data/` to the above `annotations` folder. It includes `vocab.json` and some files containing Karapthy ids.

### Training

The model is trained with default settings in the configurations file in `configs/caption/coco_config.yaml`:
The training process takes around 16 hours on a machine with 8 A100 GPU.
We also provide the code for extracting pretrained features (freezed object detector), which will speed up the training significantly.

* With default configurations (e.g., 'parallel Attention', pretrained detectors on VG or 4DS, etc):
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path

# with pretrained object detector on Visual Genome
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=vg_detector_path
```

<!-- * **More configurations will be added here for obtaining ablation results**. -->
* To freeze the backbone and detector, we can extract the region features and initial grid features first, saving it to `dataset.hdf5_path` in the config file.

**Noted that: this additional strategy will only achieve about 134 CIDEr (as reported by some researchers). To obtain 139.2 CIDEr, please train the model with freezed backbone/detector (in Pytorch, using `if 'backbone'/'detector' in n: p.requires_grad = False`) with image augmentation at every iteration. It means that we read and process every image during training rather than loading `extracted features` from hdf5.**

Then we can run the following script to train the model:
```shell
export DATA_ROOT=path/to/coco_dataset
# with pretrained object detector on 4 datasets
python train_caption.py exp.name=caption_4ds model.detector.checkpoint=4ds_detector_path \
optimizer.freezing_xe_epochs=10 \
optimizer.freezing_sc_epochs=10 \
optimizer.finetune_xe_epochs=0 \
optimizer.finetune_sc_epochs=0 \
optimizer.freeze_backbone=True \
optimizer.freeze_detector=True
```

### Evaluation

The evaluation will be run on a single GPU.
* Evaluation on **Karapthy splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```
* Evaluation on the **online splits**:
```shell
export DATA_ROOT=path/to/coco_caption
# evaluate on the validation split
python eval_caption_online.py +split='valid' exp.checkpoint=path_to_caption_checkpoint

# evaluate on the test split
python eval_caption_online.py +split='test' exp.checkpoint=path_to_caption_checkpoint
```

### Inference on RGB Image

* Perform Inference for a single image using the script `inference_caption.py`:
```
python inference_caption.py +img_path='notebooks/COCO_val2014_000000000772.jpg' \
+vocab_path='data/vocab.json' \
exp.checkpoint='path_to_caption_checkpoint'
```
*  Perform Inference for a single image using the Jupyter notebook `notebooks/Inference.ipynb`
```shell
# Require installing Jupyter(lab)
pip install jupyterlab

cd notebooks
# Open jupyter notebook
jupyter lab
```

### Finetune / Retrain GRIT on your own Dataset
We provide an example of how we finetune GRIT on the custom dataset (here is Vietnamese Image Captioning). 
Interestingly, the result shows that the GRIT checkpoint on COCO (English) benefits another language captioning task.
You may need to modify a few files only. For exapmle, we prepare 3 files in the [vicap branch](https://github.com/davidnvq/grit/tree/vicap):
* https://github.com/davidnvq/grit/blob/vicap/train_vicap.py
* https://github.com/davidnvq/grit/blob/vicap/vicap_dataset.py
* https://github.com/davidnvq/grit/blob/vicap/configs/caption/vicap_config.yaml




## Citation
If you find this code useful, please kindly cite the paper with the following bibtex:
```bibtex
@inproceedings{nguyen2022grit,
  title={Grit: Faster and better image captioning transformer using dual visual features},
  author={Nguyen, Van-Quang and Suganuma, Masanori and Okatani, Takayuki},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXVI},
  pages={167--184},
  year={2022},
  organization={Springer}
}
```

## Acknowledgement
We have inherited several open source projects into ours: i) implmentation of [Swin Transformer](https://github.com/microsoft/Swin-Transformer), ii) implementation of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and iii) implementation of image captioning base from [M2-Transformer](https://github.com/aimagelab/meshed-memory-transformer). We thank the authors of these open source projects.
