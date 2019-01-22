from setuptools import setup, find_packages
import os

setup(name='depthgen',
      version='0.2',
      description='generating depth map from single image',
      url='http://github.com/jaroslaw-weber/depthgen',
      author='Jaroslaw Weber',
      author_email='jaroslawweber@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
