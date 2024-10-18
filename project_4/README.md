# Data Science Capstone Project

This is the final project for the Data Science course for Udacity Data Science Nanodegree. I originally chose to use the source code included in the Dog Breed Classifier Workplace, but the pre-installed source code was outdated, so I planned to install a new source code based on PyTorch 2.0.

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `./dogImages`.

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `./lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

4. Create a virtual environment and install the required packages using the following command: `pip install -r requirements-all.txt`. Alternatively, you can create conda environment using the following command: `conda env create -f environment.yml`.

5. Open the notebook and follow the instructions.

```bash
jupyter notebook dog_app.ipynb
```

### Webapp

To demonstrate the model, I created a simple web app using Streamlit, which can be run using the following command:

```bash
streamlit run streamlit_app.py
```

It also was deployed on Heroku, and can be accessed at <https://udacity-data-scientist-nanodegree-hxwewfkwcgl8qykegbnqdy.streamlit.app/>

There is a screenshot of the web app:

![Screenshot](./images/Screenshot%202024-10-18%20at%2018.40.01.png)

### Project Overview

### Problem Statement

The goal of this project is to build a dog breed classifier using Convolutional Neural Networks (CNN) and transfer learning. The dataset used is the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) from Udacity.

Given an image of a dog, the model should be able to predict the breed of the dog. If the image is of a human, the model should predict the dog breed that most resembles the human.

Some images from the dataset:

![Dog](./images/Brittany_02625.jpg)

![Human](./images/sample_human_2.png)

### Metrics

The model is evaluated using the accuracy metric. The accuracy is the ratio of the number of correct predictions to the total number of predictions.

### Data

The dataset used is the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) from Udacity. The dataset contains 133 different breeds of dogs, with 6680 training images, 835 validation images, and 836 test images.

The dataset is divided into three sets: training, validation, and test. The training set is used to train the model, the validation set is used to tune the hyperparameters, and the test set is used to evaluate the model.

### Solution Statement

The solution is to build a dog breed classifier using Convolutional Neural Networks (CNN) and transfer learning. The model is trained on the training set, tuned on the validation set, and evaluated on the test set.

The model is built using PyTorch, a deep learning library for Python. PyTorch provides a flexible and dynamic computational graph, which makes it easy to build and train deep learning models.

The model is trained using transfer learning, which is a technique that allows us to use pre-trained models as a starting point. Transfer learning is useful when we have a small dataset, as it allows us to leverage the knowledge learned from a large dataset.

For this project, we use the ResNet-50 model, which is a pre-trained model that has been trained on the ImageNet dataset. The ResNet-50 model has 50 layers and is known for its performance on image classification tasks.

For detail information, please refer to the [notebook](./dog_app.ipynb).

### References

- [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [PyTorch](https://pytorch.org/)
