from setuptools import setup
import os

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ovseg',
    url='https://https://github.com/ThomasBudd/ovseg',
    author='Thomas Buddenkotte',
    author_email='tb588@cam.ac.uk',
    # Needed to actually package something
    packages=['ovseg'],
    # Needed for dependencies
    install_requires=[
            "torch>=1.7.0",
            "tqdm",
            "scikit-image>=0.14",
            "scipy",
            "numpy",
            "nibabel",
	    "rt_utils"
      ],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A deep learning based libary for ovarian cancer segmentation',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)

if 'OV_DATA_BASE' not in os.environ:
    print('\n\n OV_DATA_BASE not found! Please create the environment variable and let it point'
          'to the directory where the raw data and the trained models are.')