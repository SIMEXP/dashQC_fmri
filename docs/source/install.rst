Installation
============

We distribute  ``dashQC_fmri`` as a python package that includes both the python
modules used to re-organize your pipeline output data into the format required
by the dashboard and the javascript libraries and html templates that are used
to render these data interactively,

You can obtain ``dashQC_fmri`` either from our
`github repository <https://github.com/SIMEXP/dashQC_fmri>`_, which has the
most recent development version and is generally a good place to report and
discuss problems and extensions. Alternatively, ``dashQC_fmri`` can also be
installed via the Python Package Index.

Dependencies
------------

``dashQC_fmri`` depends on the following python packages

- numpy
- pandas
- nibabel
- pillow
- matplotlib
- nilearn
- scipy
- scikit-learn

If you install ``dashQC_fmri`` in your python environment using the methods
described below, we will take care of these dependencies for you.

If you want to run the python modules of ``dashQC_fmri`` directly without
installation, please make sure that you install these dependencies manually.

Install from source
-------------------

To install directly from our development version, run the following command::

    git clone git@github.com:SIMEXP/dashQC_fmri.git
    cd dashQC_fmri

Then you can install ``dashQC_fmri`` using::

    pip install .

If you do not have sudo permission on your system, you can also run::

    python setup.py install --user

To install ``dashQC_fmri`` in user space.

Install from ``PyPI``
---------------------

``dashQC_fmri`` will be available soon from the Python Package Index.

Running ``dashQC_fmri`` without installation
--------------------------------------------

The python modules of``dashQC_fmri`` can be run directly from the command line
without installation. Please make sure to first
install the `Dependencies`_.


