import sys
from setuptools import setup
from dashQC_fmri.__about__ import (
        __packagename__,
        __author__,
        __email__,
        __maintainer__,
        __license__,
        __description__,
        __longdesc__,
        __url__,
        CLASSIFIERS,
        REQUIRES,
        SETUP_REQUIRES,
        LINKS_REQUIRES,
        TESTS_REQUIRES,
    )


def main():
    """ Install entry-point """
    if not sys.version_info[0] == 3:
        sys.exit("\n"
                 "##########################################\n"
                 "  dashQC_fmri does not support python 2.\n"
                 "  Please install using python 3.x\n"
                 "##########################################\n")

    setup(
        name=__packagename__,
        version=0.01,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=CLASSIFIERS,
        # Dependencies handling
        setup_requires=SETUP_REQUIRES,
        install_requires=REQUIRES,
        tests_require=TESTS_REQUIRES,
        dependency_links=LINKS_REQUIRES,
        include_package_data=True,
        packages=['dashQC_fmri'],
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
