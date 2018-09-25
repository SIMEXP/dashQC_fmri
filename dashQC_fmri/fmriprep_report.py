"""

Module for generating the dashboard input data for fmriprep outputs

"""

import io
import re
import json
import time
import inspect
import argparse
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib as pal
from PIL import Image
from distutils import dir_util
from matplotlib import gridspec
from nilearn import image as ni
from nilearn import plotting as nlp
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


# TODO deal with matplotlib warnings about contour of constant values
# TODO handle session data


def find_subjects(path):
    # TODO better type checking in the find functions
    # TODO work better with relative paths (e.g. more than one level)
    if not type(path) == pal.PosixPath:
        path = pal.Path(path)
    subjects = [str(query.relative_to(path))
                for query in path.glob('sub-*') if query.is_dir()]
    return subjects


def find_runs(path, subjects):
    # TODO better type checking in the find functions
    if not type(path) == pal.PosixPath:
        path = pal.Path(path)
    runs = list()
    for sub in subjects:
        lookup_path = path / sub / 'func'
        query = lookup_path.glob('*run-*preproc.nii.gz')
        if query:
            for item in query:
                name = str(item.name).split('.')[0]
                runs.append(name)
    return runs


def populate_report(report_p):
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
        'assets/summary/js',
    ]
    for branch in tree_structure:
        branch_p = repo_p / branch
        branch_p.mkdir(parents=True, exist_ok=True)

    return


def make_str(var_name, items):
    sorted_items = sorted(items)
    item_list = [str({'id': iid + 1, 'text': item})
                 for iid, item in enumerate(sorted_items)]
    items_str = 'var {} = [\n{},\n];'.format(var_name, ',\n'.join(item_list))
    return items_str


def make_named_str(var_name, *kwargs):
    item_list = [str([i] + j) for i, j in kwargs]
    named_str = 'var {} = [{}];'.format(var_name, ',\n'.join(item_list))
    return named_str


def make_reg_anat_figure(data_in, outline_p=None, draw_outline=False):
    # TODO find a better way to deal with paths and images
    if type(data_in) == nib.Nifti1Image:
        img = data_in
    else:
        img = nib.load(str(data_in))
    if draw_outline and not outline_p:
        raise Exception('Must give path to outline image.')
    data = img.get_data()

    # Anat plot with contours
    f_anat = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 3, figure=f_anat, hspace=0)
    gs.update(bottom=0, left=0, right=1, top=1)
    ax_top = f_anat.add_subplot(gs[0, :])
    ax_mid = f_anat.add_subplot(gs[1, :])
    ax_bottom = f_anat.add_subplot(gs[2, :])
    d1 = nlp.plot_anat(img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=0, vmax=np.percentile(data, 99),
                       cut_coords=(1, 1, 1), display_mode='ortho', axes=ax_top)

    d2 = nlp.plot_anat(img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=0, vmax=np.percentile(data, 99),
                       cut_coords=(40, 30, 30), display_mode='ortho',
                       axes=ax_mid)

    d3 = nlp.plot_anat(img, annotate=False, draw_cross=False, black_bg=True,
                       vmin=0, vmax=np.percentile(data, 99),
                       cut_coords=(-40, -40, -30), display_mode='ortho',
                       axes=ax_bottom)
    if not draw_outline:
        return f_anat
    else:
        outline_img = nib.load(str(outline_p))
        d1.add_overlay(outline_img, cmap=plt.cm.coolwarm_r, alpha=0.5)
        d2.add_overlay(outline_img, cmap=plt.cm.coolwarm_r, alpha=0.5)
        d3.add_overlay(outline_img, cmap=plt.cm.coolwarm_r, alpha=0.5)

        return f_anat


def make_reg_func_figure(data_in, cmap=nlp.cm.black_red, vmin=None):
    if type(data_in) == nib.Nifti1Image:
        img = data_in
    else:
        img = nib.load(str(data_in))

    if len(img.shape) > 3:
        median = np.median(img.get_data(), 3)
        show_img = nib.Nifti1Image(median, header=img.header, affine=img.affine)
    else:
        show_img = img

    f = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 3, figure=f, hspace=0)
    gs.update(bottom=0, left=0, right=1, top=1)
    ax_top = f.add_subplot(gs[0, :])
    ax_mid = f.add_subplot(gs[1, :])
    ax_bottom = f.add_subplot(gs[2, :])
    nlp.plot_epi(show_img, annotate=False, draw_cross=False,
                 black_bg=True, bg_img=None,
                 cut_coords=(1, 1, 1), display_mode='ortho',
                 axes=ax_top, cmap=cmap, vmin=vmin,
                 colorbar=False)
    nlp.plot_epi(show_img, annotate=False, draw_cross=False,
                 black_bg=True, bg_img=None,
                 cut_coords=(40, 30, 30), display_mode='ortho',
                 axes=ax_mid, cmap=cmap, vmin=vmin,
                 colorbar=False)
    nlp.plot_epi(show_img, annotate=False, draw_cross=False,
                 black_bg=True, bg_img=None, cut_coords=(-40, -40, -30),
                 display_mode='ortho', axes=ax_bottom, cmap=cmap, vmin=vmin,
                 colorbar=False)

    return f


def target_figure(img_path, figure_path, x=3, y=1, dpi=100):
    # Ensure all paths are pathlib objects
    if not type(img_path) == pal.Path:
        img_path = pal.Path(img_path)
    if not type(figure_path) == pal.Path:
        figure_path = pal.Path(figure_path)

    raw_i = nib.load(str(img_path))
    raw = raw_i.get_data()
    # Get the median of the time points
    raw_median = np.median(raw, 3)
    raw_median_img = nib.Nifti1Image(raw_median, affine=raw_i.affine,
                                     header=raw_i.header)

    f_target = plt.figure(figsize=(x, y))
    ax = f_target.add_axes([0, 0, 1, 1])
    nlp.plot_stat_map(raw_median_img, annotate=False, bg_img=None,
                      draw_cross=False, black_bg=True, cut_coords=(1, 1, 1),
                      display_mode='ortho', axes=ax, colorbar=False)

    f_target.savefig(str(figure_path), dpi=dpi)
    # Close the figure
    plt.close(f_target)


def motion_figure(img_path, figure_path, x=3, y=1, dpi=100):
    # Ensure all paths are pathlib objects
    if not type(img_path) == pal.Path:
        img_path = pal.Path(img_path)
    if not type(figure_path) == pal.Path:
        figure_path = pal.Path(figure_path)

    # Load the image
    img = nib.load(str(img_path))
    n_t = img.shape[3]
    # Create a bytestream for the temporary files
    buf = io.BytesIO()
    byte_pos = list()
    pos = 0

    # Make the figure
    f_motion = plt.figure(figsize=(x, y))
    ax = f_motion.add_axes([0, 0, 1, 1])
    for i in np.arange(n_t, dtype=int):
        nlp.plot_stat_map(ni.index_img(img, i), annotate=False,
                          draw_cross=False, black_bg=True,
                          cut_coords=(1, 1, 1), display_mode='ortho',
                          axes=ax, colorbar=False, bg_img=None)
        f_motion.savefig(buf, dpi=dpi)
        byte_pos.append((pos, buf.tell() - pos))
        pos = buf.tell()

    plt.close(f_motion)
    # Patch them back together
    images = list()
    for i in byte_pos:
        buf.seek(i[0])
        images.append(Image.open(io.BytesIO(buf.read(i[1]))))

    new_im = Image.new('RGB', (x * dpi * n_t, y * dpi))
    for im_id, im in enumerate(images):
        new_im.paste(im, (im_id * x * dpi, 0))
    new_im.save(str(figure_path))
    # Close the stream
    buf.close()


def make_motion_grid(path):
    # TODO make useful
    # Make a time series png
    func_i = nib.load(path)

    # Find the right dimensions for the grid
    n_t = func_i.shape[-1]

    if not n_t % np.sqrt(n_t):
        x = y = np.sqrt(n_t)
    else:
        x = np.floor(np.sqrt(n_t))
        if not n_t % x:
            y = n_t / x
        else:
            y = np.floor(n_t / x) + n_t % x

    f = plt.figure(figsize=(x * 100, 1))
    gs = gridspec.GridSpec(y, 100, hspace=0, wspace=0)
    gs.update(bottom=0, left=0, right=1, top=1)
    for i in np.arange(100):
        ax = f.add_subplot(gs[i.astype(int)])
        nlp.plot_stat_map(ni.index_img(func_i, i), annotate=False,
                          draw_cross=False, black_bg=True,
                          cut_coords=(1, 1, 1), display_mode='ortho',
                          axes=ax, colorbar=False)
    return f


def make_motion_str(path):
    # This is the reference string for the .js output
    tmp_str = """var {} = {{
    columns: {},
    selection: {{ enabled: true }}, 
    onclick: function (d) {{ selectTime(d.index);}} }};"""
    ref = pd.read_csv(path, delimiter='\t')
    # Get motion information
    n_t = ref.shape[0]
    fd = ref.FramewiseDisplacement.values
    motion_mask = (fd > 0.5)
    motion_ind = np.argwhere(motion_mask).flatten()
    scrub_mask = np.zeros(n_t, dtype=np.int)
    for i in motion_ind:
        if i == 0:
            continue
        scrub_mask[i - 1:i + 3] = 1

    # Format the values for translation, rotation and motion
    tsl_txt = [
        ['motion_t{}'.format(i)] + list(ref['{}'.format(i.upper())].values)
        for i in ['x', 'y', 'z']]
    rot_txt = [
        ['motion_r{}'.format(i)] + list(ref['Rot{}'.format(i.upper())].values)
        for i in ['x', 'y', 'z']]
    # Remove the first time point. there is no FD, dashboard doesn't like nan
    # For the moment, the scrub information is also fixed to zero!
    fd_txt = [['FD'] + list(ref['FramewiseDisplacement'].values[1:]),
              ['scrub'] + list(scrub_mask)]
    out_str = '\n'.join([tmp_str.format('tsl', tsl_txt),
                         tmp_str.format('rot', rot_txt),
                         tmp_str.format('fd', fd_txt)])
    return out_str, scrub_mask, fd


def save_js(s, path):
    with open(path, 'w') as f:
        f.write(s)


def save_json(d, path):
    with open(path, 'w') as f:
        f.write(json.dumps(d, indent=4))


def make_report(prep_p, report_p, raw_p):
    if not type(report_p) == pal.Path:
        report_p = pal.Path(report_p)
    if not type(raw_p) == pal.Path:
        raw_p = pal.Path(raw_p)

    # Find the repo path where the templates are
    repo_p = pal.Path(inspect.getfile(make_report)).parents[0].absolute()

    # Hardcoded data structure for now
    # TODO read data structure from a file
    data_structure = {'chartBOLD': 'assets/summary/js/chartBOLD.js',
                      'chartBrain': 'assets/summary/js/chartBrain.js',
                      'chartT1': 'assets/summary/js/chartT1.js',
                      'fd': 'assets/summary/js/fd.js',
                      'run_motion': 'assets/motion/js/dataMotion_{}.js',
                      'listSubject': 'assets/group/js/listSubject.js',
                      'listRun': 'assets/group/js/listRun.js',
                      'fig_native_target':
                          'assets/motion/images/target_native_{}.png',
                      'fig_native_motion':
                          'assets/motion/images/motion_native_{}.png',
                      'fig_standard_target':
                          'assets/motion/images/target_stereo_{}.png',
                      'fig_standard_motion':
                          'assets/motion/images/motion_stereo_{}.png',
                      'fig_anat': 'assets/registration/images/{}_anat.png',
                      'fig_anat_raw':
                          'assets/registration/images/{}_anat_raw.png',
                      'fig_func': 'assets/registration/images/{}_func.png',
                      'fig_temp': 'assets/group/images/template_stereotaxic.png',
                      'fig_temp_raw': 'assets/group/images/template_stereotaxic_raw.png',
                      'fig_group_T1_avg': 'assets/group/images/average_t1_stereotaxic.png',
                      'fig_group_EPI_avg': 'assets/group/images/average_func_stereotaxic.png',
                      'fig_group_EPI_mask': 'assets/group/images/mask_func_group_stereotaxic.png',
                      'fig_group_EPI_mask_avg': 'assets/group/images/average_mask_func_stereotaxic.png'
                      }

    populate_report(repo_p, report_p)

    subjects = find_subjects(prep_p)
    runs = find_runs(prep_p, subjects)
    # Generate the run and subject files
    subjects_str = make_str(var_name='listSubject', items=subjects)
    save_js(subjects_str, str(report_p / data_structure['listSubject']))
    runs_str = make_str(var_name='dataRun', items=runs)
    save_js(runs_str, str(report_p / data_structure['listRun']))

    # Load the outline and the MNI template
    temp_p = repo_p / 'data/images/MNI_ICBM152_T1_asym_09c.nii.gz'
    outline_p = repo_p / 'data/images/MNI_ICBM152_09c_outline.nii.gz'
    mask_p = repo_p / 'data/images/MNI_ICBM152_09c_mask.nii.gz'
    # Create the MNI outline figure
    f_temp = make_reg_anat_figure(temp_p, outline_p, draw_outline=True)
    f_temp.savefig(report_p / data_structure['fig_temp'], dpi=200)
    f_temp_raw = make_reg_anat_figure(temp_p, draw_outline=False)
    f_temp_raw.savefig(report_p / data_structure['fig_temp_raw'], dpi=200)
    # Load the MNI data
    temp_i = nib.load(str(temp_p))
    temp_data = temp_i.get_data()
    temp_mask = nib.load(str(mask_p)).get_data().astype(bool)

    # Files created once per subject
    sub_level = {var: list() for var in ['Sub', 'T1_over', 'T1_corr',
                                         'EPI_over', 'EPI_corr']}
    anat_sum = np.zeros(temp_i.shape)
    for sid, sub in enumerate(subjects):
        sub_start = time.time()
        sub_level['Sub'].append(sub)

        # TODO make the glob pattern more flexible to deal with other spaces
        func_d = prep_p / sub / 'func'
        anat_d = prep_p / sub / 'anat'
        func_ref_p = \
            list(func_d.glob(
                '{}*MNI152NLin2009cAsym_preproc.nii.gz'.format(sub)))[
                0]
        func_mask_p = list(
            func_d.glob('{}*MNI152NLin2009cAsym_brainmask.nii.gz'.format(sub)))[
            0]
        anat_p = \
            list(anat_d.glob(
                '{}*MNI152NLin2009cAsym_preproc.nii.gz'.format(sub)))[
                0]
        anat_mask_p = list(
            anat_d.glob('{}*MNI152NLin2009cAsym_brainmask.nii.gz'.format(sub)))[
            0]

        # Get reference data
        anat = nib.load(str(anat_p)).get_data()
        anat_mask = nib.load(str(anat_mask_p)).get_data().astype(bool)
        # Func resample
        func_mask_i = nib.load(str(func_mask_p))
        func_mask_temp_i = nib.Nifti1Image(
            func_mask_i.get_data().astype(np.int), affine=func_mask_i.affine,
            header=func_mask_i.header)
        res_func_mask_i = ni.resample_img(func_mask_temp_i,
                                          target_affine=temp_i.affine,
                                          target_shape=temp_i.shape,
                                          interpolation='nearest')
        func_ref_i = nib.load(str(func_ref_p))
        func_avg_i = nib.Nifti1Image(np.mean(func_ref_i.get_data(), 3),
                                     affine=func_ref_i.affine,
                                     header=func_ref_i.header)
        res_func_ref_i = ni.resample_img(func_avg_i,
                                         target_affine=temp_i.affine,
                                         target_shape=temp_i.shape,
                                         interpolation='nearest')

        # T1 - Template
        anat_overlap = np.sum(anat_mask & temp_mask) / np.sum(anat_mask)
        sub_level['T1_over'].append(str(anat_overlap))
        anat_correlation = np.corrcoef(anat.flatten(), temp_data.flatten())[
            0, 1]
        sub_level['T1_corr'].append(str(anat_correlation))
        # Bold - T1
        func_mask = res_func_mask_i.get_data().astype(bool)
        func_ref_res = res_func_ref_i.get_data()
        func_overlap = np.sum(func_mask & anat_mask) / np.sum(func_mask)
        sub_level['EPI_over'].append(str(func_overlap))
        func_correlation = np.corrcoef(anat.flatten(), func_ref_res.flatten())[
            0, 1]
        sub_level['EPI_corr'].append(str(func_correlation))
        # Sum individual T1
        anat_sum += anat

        # Make the anat figure
        f_anat = make_reg_anat_figure(str(anat_p), outline_p, draw_outline=True)
        f_raw = make_reg_anat_figure(str(anat_p), outline_p, draw_outline=False)
        f_anat.savefig(report_p / data_structure['fig_anat'].format(sub),
                       dpi=200)
        plt.close(f_anat)
        f_raw.savefig(report_p / data_structure['fig_anat_raw'].format(sub),
                      dpi=200)
        plt.close(f_raw)
        # Make the func figure
        f_func = make_reg_func_figure(str(func_ref_p))
        f_func.savefig(report_p / data_structure['fig_func'].format(sub),
                       dpi=200)
        plt.close(f_func)
        print('Subject {}/{} done. Took {:.2f}s.'.format(sid + 1, len(subjects),
                                                         time.time() - sub_start))
    # Make T1 average
    anat_avg = anat_sum / len(subjects)
    anat_avg_i = nib.Nifti1Image(anat_avg, affine=temp_i.affine,
                                 header=temp_i.header)
    # Make the average figure
    f_anat_avg = make_reg_anat_figure(anat_avg_i)
    f_anat_avg.savefig(report_p / data_structure['fig_group_T1_avg'], dpi=200)

    # One time files
    # TODO figure out a way to deal with need for reference EPI
    bold_str = make_named_str('dataBOLD', ('Subject', subjects),
                              ('corr_target', sub_level['EPI_corr']))
    brain_str = make_named_str('dataBrain', ('Subject', subjects),
                               ('overlap_brain', sub_level['EPI_over']))
    chart_bold_str = '\n'.join([bold_str, brain_str])
    save_js(chart_bold_str, str(report_p / data_structure['chartBOLD']))

    # Make chartBrain.js - for now with dummy values
    bold_intra_str = make_named_str('dataIntra', ('Run', runs),
                                    ('corr_target', [1 for run in runs]))
    save_js(bold_intra_str, str(report_p / data_structure['chartBrain']))

    # Make T1chart.js
    t1_str = make_named_str('dataT1', ('Subject', subjects),
                            ('corr_target', sub_level['T1_corr']))
    t1_overlap_str = make_named_str('dataOverlapT1', ('Subject', subjects),
                                    ('overlap_brain', sub_level['T1_over']))
    chart_t1_str = '\n'.join([t1_str, t1_overlap_str])
    save_js(chart_t1_str, str(report_p / data_structure['chartT1']))

    # Files generated once per run
    # TODO find a better way to get the scrub mask than from the string generator
    run_level = {var: list() for var in ['Run', 'FD_before', 'FD_after',
                                         'VOL_OK', 'VOL_scrubbed']}
    for rid, run in enumerate(runs):
        run_start = time.time()
        run_name = re.search(r'.*(?=_space-MNI152NLin2009cAsym_preproc)',
                             run).group()
        sub_name = re.search(r'.*(?=_task)', run_name).group()
        run_level['Run'].append(run_name)

        run_path = prep_p / sub_name / 'func' / '{}.nii.gz'.format(run)
        run_mask_path = prep_p / sub_name / 'func' / '{}.nii.gz'.format(
            re.sub('_preproc', '_brainmask', run))
        # Group mask - we are resampling here because not all EPI sequences necessarily have the same acquisition matrix
        func_i = nib.load(str(run_path))
        func_avg_i = nib.Nifti1Image(np.mean(func_i.get_data(), 3),
                                     affine=func_i.affine, header=func_i.header)
        res_func_i = ni.resample_img(func_avg_i, target_affine=temp_i.affine,
                                     target_shape=temp_i.shape,
                                     interpolation='nearest')

        func_mask_i = nib.load(str(run_mask_path))
        func_mask_temp_i = nib.Nifti1Image(
            func_mask_i.get_data().astype(np.int), affine=func_mask_i.affine,
            header=func_mask_i.header)
        func_mask_i = ni.resample_img(func_mask_temp_i,
                                      target_affine=temp_i.affine,
                                      target_shape=temp_i.shape,
                                      interpolation='nearest')

        run_avg = res_func_i.get_data()
        run_mask = func_mask_i.get_data().astype(int)
        if rid == 0:
            group_sum = np.zeros(temp_i.shape)
            mask_sum = np.zeros(temp_i.shape)
        group_sum += run_avg
        mask_sum += run_mask

        n_t = nib.load(str(run_path)).shape[3]
        confound_path = prep_p / sub_name / 'func' / '{}_confounds.tsv'.format(
            run_name)
        # Reconstruct the name of the original run
        run_raw_path = raw_p / sub_name / 'func' / '{}.nii.gz'.format(run_name)
        if not run_path.exists():
            print(' is missing'.format(run_raw_path))
        # Make the native figures
        target_figure(str(run_raw_path),
                      report_p /
                      data_structure['fig_native_target'].format(run))
        motion_figure(str(run_raw_path), report_p /
                      data_structure['fig_native_motion'].format(run))
        # Make the standard space figure
        target_figure(str(run_path),
                      report_p / data_structure['fig_standard_target'].format(
                          run))
        motion_figure(str(run_path), report_p /
                      data_structure['fig_standard_motion'].format(run))

        # Make the dataMotion file
        data_motion_str, scrub_mask, fd = make_motion_str(str(confound_path))
        run_level['FD_before'].append(str(np.nanmean(fd)))
        run_level['FD_after'].append(str(np.nanmean(fd[~scrub_mask])))
        run_level['VOL_OK'].append(str(np.sum(~scrub_mask)))
        run_level['VOL_scrubbed'].append(str(np.sum(scrub_mask)))
        save_js(data_motion_str, str(report_p /
                                     data_structure['run_motion'].format(run)))
        print(
            'Run {}/{} done. Took {:.2f}s. {} ts: {}'.format(rid + 1, len(runs),
                                                             time.time() - run_start,
                                                             n_t, run))
    # Make the FD file
    fd_str = make_named_str('dataFD', ('Run', run_level['Run']),
                            ('FD_before', run_level['FD_before']),
                            ('FD_after', run_level['FD_after']))
    nb_str = make_named_str('dataNbVol', ('Run', run_level['Run']),
                            ('vol_scrubbed', run_level['VOL_scrubbed']),
                            ('vol_ok', run_level['VOL_OK']))
    fd_str = '\n'.join([fd_str, nb_str])
    save_js(fd_str, str(report_p / data_structure['fd']))

    # Make group level figures
    group_avg = group_sum / (rid + 1)
    group_avg_lower_cutoff = np.percentile(group_avg, 1)
    group_avg_i = nib.Nifti1Image(group_avg, affine=temp_i.affine,
                                  header=temp_i.header)
    mask_avg = mask_sum / (rid + 1)
    mask_avg_i = nib.Nifti1Image(mask_avg, affine=temp_i.affine,
                                 header=temp_i.header)
    group_mask = mask_avg > 0.95
    group_mask_i = nib.Nifti1Image(group_mask, affine=temp_i.affine,
                                   header=temp_i.header)

    f_group_avg = make_reg_func_figure(group_avg_i, vmin=group_avg_lower_cutoff)
    f_group_mask_avg = make_reg_func_figure(mask_avg_i)
    f_group_mask = make_reg_func_figure(group_mask_i, vmin=0)

    f_group_avg.savefig(report_p / data_structure['fig_group_EPI_avg'], dpi=200)
    f_group_mask_avg.savefig(report_p / data_structure['fig_group_EPI_mask_avg'],
                             dpi=200)
    f_group_mask.savefig(report_p / data_structure['fig_group_EPI_mask'],
                         dpi=200)
    return 'OK'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("preproc_dir", type=str,
                        help="bids conform directory of the fmriprep outputs")
    parser.add_argument("report_dir", type=str,
                        help="desired path for the report output")
    parser.add_argument("raw_dir", type=str,
                        help="bids conform directory of the raw fmriprep input")
    args = parser.parse_args()

    make_report(pal.Path(args.preproc_dir),
                pal.Path(args.report_dir),
                pal.Path(args.raw_dir))
