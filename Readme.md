# Speaker Change Detection

This codebase is built upon [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) and is inspired by the [HHousen/speaker-change-detection](https://github.com/HHousen/speaker-change-detection) GitHub repository. It includes modifications as presented in the paper "End-to-end speaker segmentation for overlap-aware resegmentation". The changes have been employed during the implementation of Bi-LSTM models for an empirical study in the dissertation project titled "Speaker Change Detection in a Transcript".

In addition to the Bi-LSTM models, this repository provides an unsupervised approach for speaker change detection using MFCC, KMeans, and GMM.

## Features
- **Prepare Data, Train, and Infer**: Two modes with a Bi-LSTM model:
  1. Speaker change detection.
  2. Speaker segmentation.
- **Direct Change Points Detection**: On audio files using an unsupervised method, without training.
- **Package Management**: Through Poetry.
- **Wandb.ai Integration**: For training tracking and real-time artifact saving. Ensure you have [wandb.ai](https://www.wandb.ai/) credentials for seamless code operation.

Test files and corresponding ground truth files reside in the `test_data_trimmed` directory.

## Bi-LSTM Models Setup

### Dataset
Utilized dataset: [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/). Procurement is through scripts from the [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup) GitHub repository.

### Getting Started
1. **Clone the Repository**: git clone https://github.com/rahulanand2/speaker-change-detection.git

2. **Setup**:
Navigate to the project directory and establish the environment:
* cd speaker-change-detection
* poetry install
* poetry shell

3. **Download Dataset**:
Acquire the AMI Corpus as directed in the [AMI-diarization-setup Readme](https://github.com/pyannote/AMI-diarization-setup/blob/67c2d539286e89f68952d5dcf83912bd9f01dfae/pyannote/README.md#how-to-use-in-pyannote) with `download_ami.sh`.

4. **Training**:
Start training with: python train.py

Adjust the `DO_SCD` in `train.py` to `True` for speaker change detection or `False` for segmentation.

5. **Inference**:
- Speaker change detection: `python scd_results.py`
- Segmentation: `python segmentation_results.py`

> The scripts `scd_results` and `segmentation_results` evaluate the entire test sets, recording accuracy metrics and plots to a specified local directory. Use `scd_true_final.ckpt` for `scd_results.py` and `model_segmentation.ckpt` for `segmentation_results.py`. Update the `output_dir`, `checkpoint_path`, and results directory in the respective files.

## Unsupervised Approach
Invoke the unsupervised method for speaker change detection with MFCC, BIC, Kmeans, and GMM through the Jupyter notebook, `unsupervised_scd.ipynb`. Ensure to modify the `output_dir`, `checkpoint_path`, and results paths as necessary.

## Pre-trained Model
Utilize the `scd_pretrained.ipynb` notebook on Google Colab for a pre-trained model. Access test files and save results to Google Drive by integrating the Google Colab session.

Test files have been accessed and results saved on google drive via mounting on the google colab session.