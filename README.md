A deep learning based library for segmentation of high grade serous ovarian cancer on CT images.
The library contains the code and the finals models created during my [PhD Thesis](https://doi.org/10.17863/CAM.87940).

While the code was mostly used for ovarian cancer segmentation, the library is general purpose and can be used to train other kind of segmentation models. Uncertainty Quantification based method will be added soon.

The code design is in some ways simliar to the nnU-Net library (https://github.com/MIC-DKFZ/nnUNet). Many thanks to the authors for sharing their code and letting me learn from it.

# Installation

Before you install the library make sure that your machine has a CUDA compatible GPU.
We reccommend Pascal, Turing, Volta or Ampere architecture.
To install ovseg simply clone the repo and install via pip:

```
git clone https://github.com/ThomasBudd/ovseg
cd ovseg
pip install -e .
```

Before you can run inference or training, you have to set up an environment varibale called OV_DATA_BASE.
All predictions, (pre-)trained models, raw data, etc. will be stored in this location.
If you're planning to run training on a multi-server system it is advised to set up the OV_DATA_BASE at a central location all servers can access (see run training).

# Data management

To run inference or training you first need to store the datasets in a particular way to make it accessible for the library. All datasets should be stored at $OV_DATA_BASE\raw_data and should be given a unique name. Currently the library supports datasets in which images (and segmentations) are stored as nifti or dicom files. In this current version only single channel images were tested.

If you're using **nifti files** create a folder called 'images' in $OV_DATA_BASE\raw_data\DATASET_NAME and simply put all images in there. In the case of training create a second folder called 'labels' with the corresponding segmentations. The segmentation files should have the same names as the image files or follow the Medical Decathlon naming convention (image: case_xyz_0000.nii.gz, seg: case_xyz.nii.gz). For example

	OV_DATA_BASE/raw_data/DATASET_NAME/
	├── images
  	│   ├── case_001_0000.nii.gz
    	│   ├── case_002_0000.nii.gz
    	│   ├── case_003_0000.nii.gz
   	│   ├── ...
  	├── labels
 	│   ├── case_001.nii.gz
    	|   |── case_002.nii.gz
    	│   ├── case_003.nii.gz
    	│   ├── ...
   

For **dicom images** any type of folder structure is allowed. Make sure that only axial reconstructions are contained in your dataset, the code won't remove other types of reconstructions such as topograms or sagital slices by itself. The code also assumes that all dicoms found in one folder belong to the same reconstruction, make sure that each reconstruction is contained in a seperate folder. If you're performing training, include the segmentations as dicomrt files. Each folder with reconstruction dicoms should have exactly one additional dicomrt file with the corresponding segmentation. Missing segmentations are interpreted as empty segmentations masks (only backgorund).

Examples are

	OV_DATA_BASE/raw_data/DATASET_NAME/
   	├── patient1
    	│   ├── segmentation.dcm
    	│   ├── slice1.dcm
    	│   ├── slice2.dcm
    	│   ├── slice3.dcm
    	│   ├── ...
    	├── patient2
    	│   ├── ...
    	├── patient3
    	│   ├── ...
    	├── ...


Or

	OV_DATA_BASE/raw_data/DATASET_NAME/
   	├── patient1
    	│   ├── timepoint1
    	|   │   ├── segmentation.dcm
    	|   │   ├── slice1.dcm
    	│   |   ├── slice2.dcm
    	│   |   ├── slice3.dcm
    	│   |   ├── ...
    	│   ├── timepoint2
    	│   |   ├── ...
    	├── patient2
    	│   ├── ...
    	├── patient3
    	│   ├── ...
    	├── ...

Or a mixture of the above. Note that it is not necessary to rename your dcm files to "segmentation.dcm" or "sliceX.dcm", the library will recognise it automatically.

# Run inference

To run the inference navigate to the cloned repository and run the script 'run_inference.py' with the name of your dataset as input. For example if your dataset is called 'tst_data_name' run

> python run_inference.py tst_data_name 

By default the code will run the inference for all three models and segment all disease sites considered by this library. Optionally you can specify a subset of models to run using the --models handle. The options are the following:

- pod_om: model for main disease sites in the pelvis/ovaries and the omentum. The two sites are encoded as 9 and 1 in the predictions.
- abdominal_lesions: model for various lesions between the pelvis and diaphram. The model considers lesions in the omentum (1), right upper quadrant (2), left upper quadrant (3), mesenterium (5), left paracolic gutter (6) and right  paracolic gutter (7).
- lymph_nodes: segments disease in the lymph nodes namely infrarenal lymph nodes (13), suprarenal lymph nodes (14), supradiaphragmatic lymph nodes (15) and inguinal 
lymph nodes (17).

Any combination of the three are viable options. For example if you only want to run the segmentation of the main disease sites and lymph nodes run

> python run_inference.py tst_data_name --models pod_om lymph_nodes

At first usage the library will download the pretrained weights. This might take a few minutes.

The predictions will be exported as nifti files in a location printed at the end of the inference. Additionally the predictions will be exported in the dicomrt format if the raw data was stored in the dicom format.

# Rerun ovarian cancer segmentaiton training

Repeating ovarian cancer segmentation can be done via command line without chaning any pthon code. Before the training can be started the raw data has to be preprocessed and stored. If you're running the training on a multi-sever system it is advised to place the OV_DATA_BASE in a central storage. However, this is not a good place for preprocessed data. The preprocessed data should be kept on a fast local disk to ensure that loading times do not become a bottleneck of the training. In this case create a second environment varibale called OV_PREPROCESSED that is located on such fast local disk. If this varibale is not created, the preprocessed data will be simply stored at $OV_DATA_BASE/preprocessed.

To perform preprocessing call the script 'preprocess_ovaraian_data.py' with the name of all datasets you want to use for training as arguments. For example
> python preprocess_ovarian_data.py DATANAME1 DATANAME2

Next the training can be strated by running 'run_training.py'. The first input needed is the number of the validation fold. By default the library will split the preprocessed data using a fivefold cross-validation scheme. For an input 0,1,...,4 the training will be launched using 80% of the available data for training and 20% for validation. For inputs 5,6,... the training will use 100% of the preprocessed data for training. The type of model trained is specified via the --model input. The models have the same naming as in inference (pod_om, abdominal_lesions, lymph_nodes). The training datasets used are specified via the --trn_data input.

For example, training on 100% of the data (no validation) the model for the main two disease sites on datasets called DATANAME1 and DATANAME2 run
> python run_training.py 5 --model pod_om --trn_data DATANAME1 DATANAME2

# Runing training for new segmentaiton problems

One advantage of ovseg is that it is very simple to run training, inference and modify hyper-parameters. For this you need to set up a new training scripts such as preprocess_ovarian_data.py and run_training.py.


