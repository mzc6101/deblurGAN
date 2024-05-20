from setuptools import setup

setup(
    name='deblur-gan',
    version='0.1',
    packages=['deblurgan', 'sample'],
    install_requires=[
        'absl-py>=1.0.0',
        'bleach==1.5.0',
        'click==6.7',
        'cycler==0.10.0',
        'decorator==4.2.1',
        'futures==3.1.1',
        'h5py>=3.1.0',
        'html5lib==0.9999999',
        'imageio==2.2.0',
        'Keras>=2.1.3',
        'Markdown==2.6.11',
        'matplotlib>=3.3.0',
        'networkx==2.1',
        'numpy>=1.14.0',
        'Pillow>=8.0.0',
        'pip-autoremove==0.9.0',
        'tensorflow>=2.13.0',
        'tqdm',  # Add tqdm
    ],
)