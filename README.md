# WV-Net: A SAR Wave-mode Foundation Model
This repository contains the code and model weights accompanying AIES manuscript AIES-D-25-0003 and AGU presentation IN54B-03:
"[WV-Net: A SAR Wave-mode Foundation Model](Link to your paper on arXiv/Journal)".

## Features:
-   **Pre-trained Model Weights:** Easily download and use our trained WV-Net model.
-   **Inference Demo:** A simple Jupyter notebook demonstrating how to load the model and perform inference on sample data.

## Roadmap:
-  **Reproducible codebase:** Essential code for self-supervised training and evaluation.
-  **Additional pretrained models:** More popular vision backbones pretrained used WV-Net recipe.

## Getting started:
1) Clone repo:

    ```bash
    git clone https://github.com/hawaii-ai/WV-Net.git
    cd WV-Net
    ```

2) Set up your python environment. Setting up a virtual environment using `venv` or `conda` is recommended:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3) Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4) Install the wvnet package containing reusable code:
    ```bash
    pip install -e .
    ```

5) Navigate to the notebooks directory and open a jupyter server:
    ```bash
    cd notebooks
    jupyter notebook
    ```

## Model weights:
Pre-trained model weights are located in the model_weights/ directory. More will be added.
* wvnet_resnet50_weights.pt: standard, torchvision compatible, ResNet50 weights.