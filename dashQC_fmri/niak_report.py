#!/usr/bin/env python

""" 

Script for copying NIAK fMRI preprocessing report output into new folder structure for 2018 Simexp fMRI QC dashboard

"""

import os
import time
import json
import glob
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


def clean_folder(p_folder):

    cleaned_folder = p_folder.strip()
    if cleaned_folder[len(cleaned_folder) - 1] != os.sep:
        cleaned_folder += os.sep
    return cleaned_folder


def copy_all_files(p_src_folder_wildcard, p_dest_folder):

    print("...{0}".format(p_dest_folder))

    for file in glob.glob(p_src_folder_wildcard):
        if copy_debug:
            print("Copying {0} to {1}".format(file, p_dest_folder))
        shutil.copy(file, p_dest_folder)


def create_dataset_ids(p_dataset_id_folder):

    # Date and/or timestamp will be used by dashQC as unique identifiers for a data set
    # The assumption here is that it is highly unlikely that two data sets will conflict
    # based on this distinction criteria
    with open(p_dataset_id_folder + os.sep + "datasetID.js", "w") as data_id_file:
        data_id_json = { "date": time.strftime("%Y-%m-%d-%H:%M:%S"),
                         "timestamp": time.time() }
        data_id_file.write("var datasetID = " + json.dumps(data_id_json) + ";")


def create_folder(p_new_folder):

    if not os.path.exists(p_new_folder):
        os.makedirs(p_new_folder)


def make_report(preproc_dir, report_dir):

    print("Conversion of old NIAK QC dashboard folder structure to new one commencing...")

    # (1) In output folder create the following folders:
    print("Creating new folder structure in {0}...".format(report_dir))
    populate_report(report_dir)

    # (2) Copy files from old folder structure into new one

    print("Copying files...")
    # group/*.png -> assets/group/images
    # group/*.js -> assets/group/js
    copy_all_files(preproc_dir + "group{0}*.png".format(os.sep),
                   report_dir + "assets{0}group{0}images".format(os.sep))
    copy_all_files(preproc_dir + "group{0}*.js".format(os.sep),
                   report_dir + "assets{0}group{0}js".format(os.sep))

    # motion/*.html -> assets/motion/html
    # motion/*.png -> assets/motion/images
    # motion/*.js -> assets/motion/js
    copy_all_files(preproc_dir + "motion{0}*.html".format(os.sep),
                   report_dir + "assets{0}motion{0}html".format(os.sep))
    copy_all_files(preproc_dir + "motion{0}*.png".format(os.sep),
                   report_dir + "assets{0}motion{0}images".format(os.sep))
    copy_all_files(preproc_dir + "motion{0}*.js".format(os.sep),
                   report_dir + "assets{0}motion{0}js".format(os.sep))

    # qc_registration.csv -> assets/registration/csv
    # registration/*.png -> assets/registration/images
    copy_all_files(preproc_dir + "qc_registration.csv".format(os.sep),
                   report_dir + "assets{0}registration{0}csv".format(os.sep))
    copy_all_files(preproc_dir + "registration{0}*.png".format(os.sep),
                   report_dir + "assets{0}registration{0}images".format(os.sep))

    # summary/*.js -> assets/summary/js
    copy_all_files(preproc_dir + "summary{0}*.js".format(os.sep),
                   report_dir + "assets{0}summary{0}js".format(os.sep))

    # (3) Create a JSON file for this conversion session that registers this as a unique data set for the dashQC

    print("Creating unique IDs for this data set...")
    create_dataset_ids(report_dir + "assets{0}registration{0}js".format(os.sep))
 
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
