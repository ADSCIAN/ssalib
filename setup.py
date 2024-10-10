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
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'full': ['dask']
    }
)
