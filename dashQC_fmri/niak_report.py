#!/usr/bin/env python

""" 

Script for copying NIAK fMRI preprocessing report output into new folder structure for 2018 Simexp fMRI QC dashboard

"""

import time
import json
import shutil
import inspect
import argparse
import pathlib as pal
from distutils import dir_util

copy_debug = False


def populate_report(report_p):

    if not type(report_p) == pal.Path:
        report_p = pal.Path(report_p)
    # Copy the template into the report folder
    repo_p = pal.Path(inspect.getfile(make_report)).parents[0].absolute()
    dir_util.copy_tree(str(repo_p / 'data/report'), str(report_p), verbose=0)
    # Create the directory tree for the files that are yet to be created
    tree_structure = [
        'assets/group/images',
        'assets/group/js',
        'assets/motion/images',
        'assets/motion/js',
        'assets/registration/images',
        'assets/registration/js',
        'assets/summary/js',
    ]
    for branch in tree_structure:
        branch_p = report_p / branch
        branch_p.mkdir(parents=True, exist_ok=True)

    return


def copy_all_files(p_src_folder_wildcard, p_dest_folder):

    print("...{0}".format(p_dest_folder))

    for file in p_src_folder_wildcard.parent.glob(p_src_folder_wildcard.name):
        if copy_debug:
            print("Copying {0} to {1}".format(file, p_dest_folder))
        shutil.copy(file, p_dest_folder)


def create_dataset_ids(p_dataset_id_folder):

    # Date and/or timestamp will be used by dashQC as unique identifiers for a data set
    # The assumption here is that it is highly unlikely that two data sets will conflict
    # based on this distinction criteria
    with (p_dataset_id_folder / "datasetID.js").open("w") as data_id_file:
        data_id_json = { "date": time.strftime("%Y-%m-%d-%H:%M:%S"),
                         "timestamp": time.time() }
        data_id_file.write("var datasetID = " + json.dumps(data_id_json) + ";")


def make_report(preproc_dir, report_dir):
    if not issubclass(type(preproc_dir), pal.Path):
        preproc_dir = pal.Path(preproc_dir)
    if not issubclass(type(report_dir), pal.Path):
        report_dir = pal.Path(report_dir)

    print("Conversion of old NIAK QC dashboard folder structure to new one commencing...")

    # (1) In output folder create the following folders:
    print("Creating new folder structure in {0}...".format(report_dir))
    populate_report(report_dir)

    # (2) Copy files from old folder structure into new one

    print("Copying files...")
    # group/*.png -> assets/group/images
    # group/*.js -> assets/group/js
    copy_all_files(preproc_dir / "group/*.png",
                   report_dir / "assets/group/images")
    copy_all_files(preproc_dir / "group/*.js",
                   report_dir / "assets/group/js")

    # motion/*.html -> assets/motion/html
    # motion/*.png -> assets/motion/images
    # motion/*.js -> assets/motion/js
    #copy_all_files(preproc_dir + "motion{0}*.html".format(os.sep),
    #               report_dir + "assets{0}motion{0}html".format(os.sep))
    copy_all_files(preproc_dir / "motion/*.png",
                   report_dir / "assets/motion/images")
    copy_all_files(preproc_dir / "motion/*.js",
                   report_dir / "assets/motion/js")

    # qc_registration.csv -> assets/registration/csv
    # registration/*.png -> assets/registration/images
    copy_all_files(preproc_dir / "qc_registration.csv",
                   report_dir / "assets/registration/csv")
    copy_all_files(preproc_dir / "registration/*.png",
                   report_dir / "assets/registration/images")

    # summary/*.js -> assets/summary/js
    copy_all_files(preproc_dir / "summary/*.js",
                   report_dir / "assets/summary/js")

    # (3) Create a JSON file for this conversion session that registers this as a unique data set for the dashQC

    print("Creating unique IDs for this data set...")
    create_dataset_ids(report_dir / "assets/registration/js")
 
    print("Conversion complete.")


if "__main__" == __name__:

    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_dir", type=str,
                        help="path to the dire/mnt/data_sq/cisl/surchs/ABIDE/ABIDE_1/PREPROCESS_NIAK/NYUctory with the niak preprocessed data")
    parser.add_argument("report_dir", type=str,
                        help="desired path for the report output")
    args = parser.parse_args()

    # We want to point at the report folder in the niak preprocessing root
    preproc_p = pal.Path(args.preproc_dir)
    if str(preproc_p).endswith('report'):
        make_report(args.preproc_dir, args.report_dir)
    elif (preproc_p / 'report').exists():
        make_report(str(preproc_p / 'report'), args.report_dir)
    else:
        # It's probably an invalid path but we'll let it error out later down the line
        make_report(args.preproc_dir, args.report_dir)
