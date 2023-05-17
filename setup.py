__author__ = 'Anonymous'


from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

    
setup(
    name='adversarial sensitivity',
    version='1.0.0',
    author='Anonymous',
    author_email='Anonymous@gmail.com',
    description='Description of your package',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)