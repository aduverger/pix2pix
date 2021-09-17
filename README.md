# pix2pix

Image-to-image translation using conditional generative adversarial networks (cGAN).

The objective was to reproduce the [original research paper](https://arxiv.org/abs/1611.07004) from Isola et al. to generate realistic building facades from sketches.

![Generated image](https://aduverger.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff80adc2b-9f33-4f35-8030-0eacdb2e2a77%2Ftest2.jpg?table=block&id=36acfc6b-6b78-4722-8905-2486c407cfb6&spaceId=b9592099-2b37-4101-aeaa-24da873f1526&width=2880&userId=&cache=v2 "Generated image")

A website is available to try the fitted model, by uploading sketches or draw ones yourself : [yeswegan.herokuapp.com](https://yeswegan.herokuapp.com/)


# Put the API in prod

You need to download a `gen_pix2pix_400_model_save.h5` model at the root of the project
