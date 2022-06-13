# ovseg
A deep learning based library for segmentation of high grade serous ovarian cancer on CT images.

# Installation

Before you install the library make sure that your machine has a CUDA compatible GPU.
We reccommend Pascal, Turing, Volta or Ampere architecture.
To install ovseg simply clone the repo and install via pip:

```
git clone https://github.com/ThomasBudd/ovseg
cd ovseg
pip instal -e .
```

Before you can run inference or training, you have to set up an environment varibale called OV_DATA_BASE.
All predictions, (pre-)trained models, raw data, etc. will be stored in this location.
If you're planning to run training on a multi-server system it is advised to set up the OV_DATA_BASE at a central location all servers can access (see run training).

# Run inference

# Run training

