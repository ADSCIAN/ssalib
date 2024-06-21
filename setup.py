from setuptools import setup, find_packages

setup(
    name='vassal',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'vassal': ['datasets/*.txt', 'datasets/*.csv', 'datasets/*.json']
    },
    url='https://github.com/ADSCIAN/vassal',
    license='BSD-3-Clause',
    author='Damien Delforge',
    author_email='damien.delforge@adscian.be',
    description='Visual and Automated Singular Spectrum Analysis Library',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
    ],
    extras_require={
        'full': ['scipy', 'scikit-learn', 'dask']
    }
)
