from setuptools import setup, find_packages

setup(
    name='ebbpy',
    version='0.1',
    description='Empirical Bayes methods for beta-binomial models in Python',
    author='Matt Savoca',
    author_email='mattscottsavoca@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'polars',
        'scipy',
        # Include other dependencies as needed
    ],
)