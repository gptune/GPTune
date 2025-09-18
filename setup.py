from setuptools import setup
import os
import io

NAME = 'GPTune'
DESCRIPTION = ''

here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = dict()
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

REQUIRED=[]


EXTRAS = {
    'all': [
        'Sphinx>=1.8.2',
        'sphinx_rtd_theme',
        'statsmodels >= 0.13.0',
        'numpy',
        'joblib',
        'scikit-learn <= 0.22',
        'scipy',
        'pyaml',
        'matplotlib',
        'GPy',
        'openturns',
        'lhsmdu',
        'ipyparallel',
        'opentuner >= 0.8.8',
        'hpbandster',
        'pygmo<=2.16.1',
        'filelock', 
        'requests',           
        'pymoo',           
        'cloudpickle',           
        'SALib',           
        'scikit-optimize @ git+https://git@github.com/gptune/scikit-optimize.git',           
        'ytopt-autotune @ git+https://git@github.com/gptune/autotune.git',           
        'cgp @ git+https://git@github.com/gptune/cGP.git',           
        'hybridMinimization @ git+https://git@github.com/gptune/hybridMinimization.git',           
    ],
}

setup(
  name=NAME,
  version=about['__version__'],
  license='BSD',
  author="Yang Liu",
  author_email='liuyangzhuan@lbl.gov',
  description=DESCRIPTION,
  packages=['GPTune'],
  url='https://github.com/gptune/GPTune',
  install_requires=REQUIRED,
  extras_require=EXTRAS,
  include_package_data=True,
  package_data = {'GPTune': ['lib_*']}
)