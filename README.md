## Dataset
Due to GitHub size limits, `cifar-10-python.tar.gz` is **not included**.  
Download it from [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in the `dataset/` folder.

# Data Augmentation Project

## Overview
This project implements various **data augmentation techniques** for images, allowing you to expand your dataset and improve the performance of machine learning models. It supports common transformations such as rotation, flipping, scaling, translation, color jitter, and more.

This repository demonstrates how to apply augmentation **programmatically** using Python and popular libraries such as **OpenCV** and **NumPy**.

---

## Features
- Rotate, flip, and crop images  
- Adjust brightness, contrast, and color  
- Add noise and blur to images  
- Batch processing for large datasets  
- Customizable augmentation pipelines  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nikshitha987/Data-Augmentation.git
cd Data-Augmentation
Create a Python virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
Activate the virtual environment:

Windows (PowerShell):

bash
Copy code
venv\Scripts\Activate.ps1
Windows (CMD):

bash
Copy code
venv\Scripts\activate
Linux / macOS:

bash
Copy code
source venv/bin/activate
Install required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Place your dataset in the dataset/ folder.

Run the augmentation script:

bash
Copy code
python augment.py
The augmented images will be saved in the output/ folder (or as configured in the script).

Dataset
Important: Large datasets are not included due to GitHub file size limits.

Example: CIFAR-10 dataset (~162 MB)

Download manually from CIFAR-10 Dataset

Place the downloaded file inside the dataset/ folder:

text
Copy code
dataset/cifar-10-python.tar.gz
Project Structure
graphql
Copy code
Data-Augmentation/
│
├── dataset/                # Input dataset (not included)
├── output/                 # Augmented images
├── augment.py              # Main augmentation script
├── requirements.txt        # Required Python packages
├── .gitignore              # Ignored files and folders
└── README.md               # Project documentatio
