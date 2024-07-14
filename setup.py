from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
long_description = None
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='vsom',  # 包名称
      packages=['vsom'],  # 需要处理的包目录
      version='0.1.0',  # 版本
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python', 'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
      ])