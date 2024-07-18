from setuptools import setup, find_namespace_packages

setup(
    name='qrdet',
    version='2.5',
    author_email='eric@ericcanas.com',
    author='Eric Canas',
    url='https://github.com/Eric-Canas/qrdet',
    description='Robust QR Detector based on YOLOv8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(),
    # expose qreader.py as the unique module
    license='MIT',
    install_requires=[
        'ultralytics',
        'quadrilateral-fitter',

        'numpy',
        'requests',
        'tqdm'
        'simpy==1.11.1'
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