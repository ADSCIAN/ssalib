from setuptools import setup, find_packages

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if
            line.strip() and not line.startswith('#')]

setup(
    name='vassal',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'vassal': ['datasets/*.txt', 'datasets/*.csv', 'datasets/*.json']
    },
    url='https://github.com/ADSCIAN/vassal',
    license='BSD-3-Clause',
    author='Damien Delforge, Alice Alonso',
    author_email='damien.delforge@adscian.be',
    description='Visual and Automated Singular Spectrum Analysis (VASSAL)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Add this for proper README rendering on PyPI
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'full': ['dask'],
        'test': [
            'pytest',
            'pytest-cov',
        ],
        'dev': [
            'pytest',
            'pytest-cov',
            'black',  # for code formatting
            'flake8',  # for code linting
            'dask',  # include optional dependency for development
        ]
    },
    python_requires='>=3.8',  # We can adjust this based on test results
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
    ],
    keywords='singular spectrum analysis, time series, decomposition',
)