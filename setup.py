from setuptools import setup, find_namespace_packages

setup(
    name='qrdet',
    version='1.9',
    packages=find_namespace_packages(),
    # expose qreader.py as the unique module
    py_modules=['qrdet'],
    url='https://github.com/Eric-Canas/qrdet',
    license='MIT',
    author='Eric Canas',
    author_email='elcorreodeharu@gmail.com',
    description='Robust QR Detector based on YOLOv7',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'yolov7-package',
        'requests',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Topic :: Multimedia :: Graphics',
        'Typing :: Typed',
    ],
)
