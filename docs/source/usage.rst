Using ``dashQC_fmri``
=====================

``dashQC_fmri`` presents your data in 3 thematically separate but interlinked
views:



Summary Page
------------

.. image:: https://github.com/SIMEXP/dashQC_fmri/raw/master/docs/source/dash_summary.jpg

Information on the dashboard's summary page is divided into several subsections accessible via the 'Registration Stats', 'Motion Stats', 'Group Images', and 'Provenance' pills.

Registration Stats
******************

The Registration Stats subsection consists of subject registration chart and table views. Data on the table is displayed on the chart above. The table itself is sortable by clicking the table column headers and searchable via the search box in the upper right of the table. Each row of the table represents a subject. Each column in the table ('T1 corr_target', 'T1 overlap_brain', 'BOLD corr_target', and "BOLD overlap_brain") corresponds to its own chart. Clicking on column headers highlights and sorts that entire column and displays the appropriate chart in the view above. Clicking on single rows highlight that subject in the chart view above as well. Similarly, single clicks on bars in the chart will highlight that subject's value for the chart. Double clicking a row or a bar in a chart will take you to the registration page for that subject.

Motion Stats
******************

The Motion Stats subsection consists of run motion chart and table views. The table works similarly to the subject table in the Registration Stats subsection, but here each row represents a run. The columns in this table, however, do not correspond to a single chart. Clicking 'corr_target' reveals a chart view of the column data, but the pairs 'FD_Before' and 'FD_After' as well as 'vol_scrubbed' and "vol_ok" are both represented on one chart each. Single clicks in the chart and table function like single clicks in Registration Stats, but double clicking a row or a bar in a Motion Stats chart will instead take you to the motion page for that run.

Group Images
******************

Moving the mouse over the four images in this subsection allows comparison between group averaged images to templates.

Provenance
******************

The Provenance subsection gives information on the overall pipeline process and options, as well as per subject information. Click the "Input files" dropdown to switch between subjects. The "Input files for subject" and "Pipeline options" views feature collapsible JSON for easier reading.


Registration Page
-----------------

.. image:: https://github.com/SIMEXP/dashQC_fmri/raw/master/docs/source/dash_registration.jpg

This page is the area where users can actively perform quality control on their subject images. Two main image views available are "Individual vs template T1 scan for <the current subject>" and "BOLD vs T1 scan for <the current subject>". These images make use of comparison via mouse over similar to the Group Images subsection on the Summary page. Subject images can be moved between by either clicking the arrow buttons on either side of the two images or using the keyboard's left and right arrow keys. Below these two images is an area where the user can mark the quality level of the scans via buttons (pass, maybe, fail, or unmarked), and can also enter comments for each subject. These ratings are also enterable via hotkeys as well (p for pass, m for maybe, f for fail, and u for unmarked). Subjects can be moved between more directly via the "Current subject" dropdown in this area. Comments and ratings for the scans of each subject are stored in the browser's web session storage. Unless the browser cache is cleared or the data set in the underlying folder is swapped out, these comments and ratings will remain stored even when the browser is closed. However, should users want to save a local JSON copy of this data, they can click the "Save QC" button in the top right of the page on the navbar. Users can also load previous ratings and comments (from a previously saved QC JSON file) on the current scans via the "Load QC" button, also on the top right of the page.


Motion Page
-----------

.. image:: https://github.com/SIMEXP/dashQC_fmri/raw/master/docs/source/dash_motion.jpg

This page shows several views of subject motion during runs of scans. Different runs are accessible via the 'Select a run' dropdown on the top right of the page. The three charts on the left feature frame displacement as well as translation and rotation parameters frame by frame. The two top images on the right display an animated view of fMRI frames in both native and stereotaxic space over time. The bottom image on the right displays the overall run versus reference volume. Frames on the top two images can be advanced via the arrow buttons as well as the left and right arrows on the keyboard. Moving between frames in this way also highlights the appropriate data point for the frame in the charts on the left. Hovering over points on the charts on the left reveal their data values and clicking on them advances the animation to the appropriate frame in the top two images.