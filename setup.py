from setuptools import setup, find_packages
import os

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ovseg',
    url='https://https://github.com/ThomasBudd/ovseg',
    author='Thomas Buddenkotte',
    author_email='thomasbuddenkotte@googlemail.com',
    # Needed to actually package something
    packages=find_packages('src'),
    package_dir={'': 'src'},
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
    entry_points={'console_scripts': ['ovseg_inference = ovseg.run.run_inference:main']},
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='A deep learning based libary for ovarian cancer segmentation',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
