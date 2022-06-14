seg
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

To run the inference you have to first copy your raw data (called tst_data from here on) to a subfolder called 'raw_data' at OV_DATA_BASE. Currently two datatypes for raw data are accepted: nifti and dicom.

If you're using nifti files create a folder called 'images' in $OV_DATA_BASE\raw_data\tst_data and simply put all images in there.
For dicom images any type of folder structure is allowed. Make sure that only axial reconstructions are contained in your test data, the code won't remove other types of reconstructions such as topograms or sagital slices by itself. The code also assumes that all dicoms found in one folder belong to the same reconstruction, make sure that each reconstruction is contained in a seperate folder.

To run the inference navigate to the cloned repository and run the script 'run_inference.py' with the name of your dataset (here tst_data) as input.
By default the code will run the inference for all three models and segment all disease sites considered by this library. Optionally you can specify a subset of models to run using the --models handle. The options are the following:

- pod_om: model for main disease sites in the pelvis/ovaries and the omentum. The two sites are encoded as 9 and 1.
- abdominal_lesions: model for various lesions between the pelvis and diaphram. The model considers lesions in the omentum (1), right upper quadrant (2), left upper quadrant (3), mesenterium (5), left paracolic gutter (6) and right  paracolic gutter (7).
- lymph_nodes: segments disease in the lymph nodes namely infrarenal lymph nodes (13), suprarenal lymph nodes (14), supradiaphragmatic lymph nodes (15) and inguinal 
lymph nodes (17).

Any combination of the three are viable options. For example if you only want to run the segmentation of the main disease sites and lymph nodes run

python run_inference.py tst_data --models pod_om lymph_nodes

# Run training

