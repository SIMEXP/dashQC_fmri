"""For having the version."""

import pkg_resources

__version__ = pkg_resources.require("dashQC_fmri")[0].version
