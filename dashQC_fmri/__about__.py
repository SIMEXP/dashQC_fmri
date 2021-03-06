__author__ = 'The dashQC_fmri developers'
__copyright__ = 'Copyright 2018, The SIMEXP lab'
__credits__ = [
    'Jonathan Armoza', 'Pierre Bellec', 'Yassine Benhajali',  'Sebastian Urchs'
]
__license__ = 'MIT'
__maintainer__ = 'Sebastian Urchs'
__email__ = 'sebastian.urchs@mail.mcgill.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/SIMEXP/bids_qcdash'
__packagename__ = 'dashQC_fmri'
# TODO update description
__description__ = ("dashQC_fmri is an interactive quality control dashboard that facilitates "
                   "the manual rating of resting state fMRI focused preprocessing.")
# TODO long description
__longdesc__ = """\
Coming soon.
"""

SETUP_REQUIRES = [
    'setuptools>=18.0',
]

REQUIRES = [
    'numpy',
    'pandas',
    'nibabel',
    'pillow',
    'matplotlib',
    'nilearn',
    'scipy',
    'scikit-learn',
    'imageio'
]

LINKS_REQUIRES = [
]

TESTS_REQUIRES = [
]

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Analysis',
    'License :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]
