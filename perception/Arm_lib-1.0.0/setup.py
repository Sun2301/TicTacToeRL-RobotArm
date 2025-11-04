"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='Arm_lib',
    version='1.0.0',
    author='Hanson',
    author_email='821058392@qq.com',
    description='Arm_lib模块',
    long_description=long_description,  # 这里是文档内容, 读取readme文件
    long_description_content_type='text/markdown',  # 文档格式
    url='https://www.haisheng666.xin/user_system/login?next=/robot/',  # 这里是个人对话机器人展示
    packages=find_packages(),
    classifiers=[  # 这里我们指定证书, python版本和系统类型
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',  # 这里指定python版本号必须大于3.7才可以安装
    # install_requires=['requests']  # 我们的模块所用到的依赖, 这里指定的话, 用户安装你的模块时, 会自动安装这些依赖
)
