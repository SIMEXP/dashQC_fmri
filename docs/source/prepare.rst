Preparing data for ``dashQC_fmri``
==================================

To run ``dashQC_fmri`` on your own preprocessed data from
`fmriprep <https://fmriprep.readthedocs.io/en/stable/>`_ or
`NIAK <http://niak.simexp-lab.org/>`_ you need to first run the corresponding
data converter module. The converter modules are python scripts that reorganize
the output data from ``fmriprep`` or ``NIAK`` into the folder structure expected
by ``dashQC_fmri``. They also pregenerate some of the larger visual
representations so you can later run the dashboard without any lag. This process
can take some time but only has to be performed once per preprocessed dataset.

.. note:: Make sure that you have completed the :ref:`install:Installation`  of ``dashQC_fmri`` before you try to follow the steps in this section.

Preparing data from a ``fmriprep`` pipeline
-------------------------------------------

.. note:: To generate the ``dashQC_fmri`` data for a ``fmriprep`` pipeline we currently need access to the unprocessed raw data folder on which ``fmriprep`` was run. Please make sure that this location is reachable before you run the converter.

Let's assume the raw data folder that ``fmriprep`` was run on is located at
:code:`/path/to/my/raw/data` and the output directory of your successfully
completed `fmriprep`pipeline is located at :code:`/path/to/my/fmriprep/output/`.
To generate your ``dashQC_fmri`` report at :code:`/path/to/my/dashQC_fmri/report`
you would then run

    python -m dashQC_fmri.fmriprep_report /path/to/my/fmriprep/output/ /path/to/my/dashQC_fmri/report /path/to/my/raw/data

Once the process has completed, navigate to :code:`/path/to/my/dashQC_fmri/report`
e.g.:

    cd /path/to/my/dashQC_fmri/report

find the :code:`index.html` file and open it in a browser (e.g. on Unix):

    sensible-browser index.html

This will open the ``dashQC_fmri`` dashboard you have just generated.


Preparing data from a ``niak`` pipeline
---------------------------------------

Let's assume the output directory of your successfully
completed `NIAK` pipeline is located at :code:`/path/to/my/niak/output/`.
To generate your ``dashQC_fmri`` report at :code:`/path/to/my/dashQC_fmri/report`
you would then run

    python -m dashQC_fmri.niak_report /path/to/my/niak/output/ /path/to/my/dashQC_fmri/report /path/to/my/raw/data

Once the process has completed, navigate to :code:`/path/to/my/dashQC_fmri/report`
e.g.:

    cd /path/to/my/dashQC_fmri/report

find the :code:`index.html` file and open it in a browser (e.g. on Unix):

    sensible-browser index.html

This will open the ``dashQC_fmri`` dashboard you have just generated.

