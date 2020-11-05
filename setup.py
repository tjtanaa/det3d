from setuptools import setup, find_packages

setup(name='det3d',
      version='0.0.1',
      # list folders, not files
      packages=find_packages(exclude=['det3d.tests'])
      )