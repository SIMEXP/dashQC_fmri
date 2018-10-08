#!/usr/bin/env python

""" 

Script for copying NIAK fMRI preprocessing report output into new folder structure for 2018 Simexp fMRI QC dashboard

"""

import os
import glob
import shutil
import argparse

copy_debug = False


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


def create_folder(p_new_folder):
    if not os.path.exists(p_new_folder):
        os.makedirs(p_new_folder)


def make_report(preproc_dir, report_dir):
    print("Conversion of old NIAK QC dashboard folder structure to new one commencing...")

    # (1) In output folder create the following folders:
    print("Creating new folder structure in {0}...".format(report_dir))

    # assets
    create_folder(report_dir + "assets")

    # assets/group
    # assets/group/images
    # assets/group/js
    create_folder(report_dir + "assets{0}group".format(os.sep))
    create_folder(report_dir + "assets{0}group{0}images".format(os.sep))
    create_folder(report_dir + "assets{0}group{0}js".format(os.sep))

    # assets/motion
    # assets/motion/html
    # assets/motion/images
    # assets/motion/js
    create_folder(report_dir + "assets{0}motion".format(os.sep))
    create_folder(report_dir + "assets{0}motion{0}html".format(os.sep))
    create_folder(report_dir + "assets{0}motion{0}images".format(os.sep))
    create_folder(report_dir + "assets{0}motion{0}js".format(os.sep))

    # assets/registration
    # assets/registration/csv
    # assets/registration/images
    create_folder(report_dir + "assets{0}registration".format(os.sep))
    create_folder(report_dir + "assets{0}registration{0}csv".format(os.sep))
    create_folder(report_dir + "assets{0}registration{0}images".format(os.sep))

    # assets/summary
    # assets/summary/js
    create_folder(report_dir + "assets{0}summary".format(os.sep))
    create_folder(report_dir + "assets{0}summary{0}js".format(os.sep))

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
    
    print("Conversion complete.")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_dir", type=str,
                        help="path to directory with niak report raw data")
    parser.add_argument("report_dir", type=str,
                        help="desired path for the report output")
    args = parser.parse_args()
    make_report(args.preproc_dir, args.report_dir)


