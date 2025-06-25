from setuptools import setup, find_packages

setup(
    name='ssalib',
    version='0.1.2b1',
    packages=find_packages(),
    package_data={
        'ssalib': ['datasets/*.txt', 'datasets/*.csv', 'datasets/*.json']
    },
    url='https://github.com/ADSCIAN/ssalib',
    license='BSD-3-Clause',
    author='Damien Delforge, Alice Alonso, Oliver de Viron, Marnik Vanclooster, Niko Speybroeck',
    author_email='damien.delforge@adscian.be',
    description='Singular Spectrum Analysis Library (SSALib)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'joblib',
        'numpy',
        'matplotlib',
        'pandas',
        'scipy<1.16.0',
        'scikit-learn',
        'statsmodels'
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'flake8'
        ]
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering',
    ],
    keywords='singular spectrum analysis, time series, decomposition',
)
