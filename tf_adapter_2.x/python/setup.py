from setuptools import setup, Extension
from setuptools import find_packages

setup(name='npu_device',
      version='0.1',
      description='This is a demo package',
      long_description='This is a demo package',
      packages=find_packages(),
      include_package_data=True,
      ext_modules=[],
      zip_safe=False)
