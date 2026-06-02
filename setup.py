from setuptools import setup, find_packages

setup(
    name='ml-mini-projects',
    version='1.0.0',
    author='Bharat Tomar',
    author_email='bharat@akgec.ac.in',
    description='Python ML notebooks and reusable utilities — data preprocessing, classification, regression, clustering & neural networks',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24',
        'pandas>=2.0',
        'scikit-learn>=1.3',
        'matplotlib>=3.7',
        'seaborn>=0.12',
        'torch>=2.1',
        'joblib>=1.3',
        'tqdm>=4.66',
    ],
)
