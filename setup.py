#!/usr/bin/env python

from setuptools import setup


setup(
    name="boundary_iou",
    packages=["boundary_iou"],
    package_dir = {"boundary_iou": "boundary_iou"},
    version="0.1",
    author="bowencheng",
    python_requires=">=3.6",
    install_requires=[
        "setuptools",
        "cython",
        "numpy",
        "tabulate",
        "fvcore>=0.1.1",
        "pycocotools>=2.0.1",
        "cityscapesscripts>=1.5.0",
        "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip"
    ],
)
