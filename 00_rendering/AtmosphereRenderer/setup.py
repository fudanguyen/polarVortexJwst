from setuptools import setup, find_packages

setup(
    name='AtmosphereRenderer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A library for rendering and generating atmospheric models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/AtmosphereRenderer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pyvista',
        'tqdm',
        'h5py',
        'Pillow',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)