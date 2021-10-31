# ele.gan.t facades

Try our notebook: &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aduverger/pix2pix/blob/master/notebooks/elegant_facades.ipynb)

Image-to-image translation using conditional generative adversarial networks (cGAN).

The objective was to reproduce the [original research paper](https://arxiv.org/abs/1611.07004) from Isola et al. to generate realistic building facades from sketches.

<p align="center">
  <img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff80adc2b-9f33-4f35-8030-0eacdb2e2a77%2Ftest2.jpg?table=block&id=36acfc6b-6b78-4722-8905-2486c407cfb6&spaceId=b9592099-2b37-4101-aeaa-24da873f1526&width=2000&userId=46c52212-dcdf-44a2-b4b8-796d09177007&cache=v2" />
</p>
<p align="center">
Generated image from the cGan model
  </p>

A website is available to try the fitted model, by uploading sketches or draw ones yourself : [yeswegan.herokuapp.com](https://yeswegan.herokuapp.com/)


# Setup

## Install the package

The easiest way to use this library is through the Colab Notebook provided on top of this page.
However, if you prefer to run this library outside a notebook, please follow the steps below.

Clone the project and install it:

```bash
git clone git@github.com:aduverger/pix2pix.git
cd pix2pix
pip install -r requirements.txt
make clean install test         # install and test
```

## Download the data

You can download the facades dataset from Berkeley website:

```bash
cd datasets
curl -O  http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
tar -xzf facades.tar.gz && rm facades.tar.gz
cd ..
```

## Train a model

```bash
pix2pix-train --model_name MY_MODEL
```

Alternatively, you have access to some hyperparameters to train a model.
Please refer to the documentation of the .fit() method for details about these parameters.

```bash
pix2pix-train --model_name MY_MODEL --init 0 --epochs 200 --epochs_gen 5 --epochs_disc 0 --k 1 --l1_lambda 100
```

## Generate a facade

```bash
pix2pix-predict --model_name MY_MODEL
```

Alternatively, you can choose the sketch image you want your model to generate a facade from.

You can choose a sketch by specifying an index from the test sample:

```bash
pix2pix-predict --model_name MY_MODEL --test_index 89
```
Or you can use a sketch of your own by specifying its file path:
```bash
pix2pix-predict --model_name MY_MODEL --file_path /~/Downloads/some_sketch.jpg
```

