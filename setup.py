from setuptools import setup, find_packages
from odk import __version__ as version

setup(
    name='odk',
    version=version,
    packages= find_packages(),
    description="Replace PyTorch's internal operators with your custom Triton kernels",
    author='Milad Rahimi',
    author_email='miladrahimipo@gmail.com',
    url='https://github.com/milad2073/open-deep-kernel',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch',
        'triton',
    ],
)