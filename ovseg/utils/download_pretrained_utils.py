import requests
import os
import zipfile
from tqdm import tqdm

def download_and_install(url):
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    local_filename = os.path.join(os.environ['OV_DATA_BASE'], 'temp.zip')
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192 * 16)): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk: 
                f.write(chunk)
                
    # extracting the zip and removing zip file
    # borrowed from nnUNet: 
    # https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/inference/pretrained_models/download_pretrained_model.py#L294
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(os.environ['OV_DATA_BASE'])
    
    if os.path.isfile(local_filename):
        os.remove(local_filename)


def maybe_download_clara_models():
    
    if 'OV_DATA_BASE' not in os.environ:
        raise FileNotFoundError('Environment variable \'OV_DATA_BASE\' was not set.'
                                'Please do so to specify where pretrained models, raw data'
                                'and predictions should be stored.')

    if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')):
        
        print('Downloading pretrained models (4080 chunks)...')
        
        # url = "https://sandbox.zenodo.org/record/1071186/files/clara_models.zip?download=1"
        url = "https://sandbox.zenodo.org/record/33549/files/clara_models.zip?download=1"
        download_and_install(url)
        print('Done!')

