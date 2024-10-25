import numpy as np
import copy
import mne
from mne.coreg import Coregistration
from mne.io import read_info
import os

# See https://mne.tools/1.7/auto_tutorials/forward/25_automated_coreg.html#tut-auto-coreg

subjectID = 'mnsbp001'

# Files to read
dataDir = '../'
fifName = 'MNS_shortISI_preprocessed_reg-raw.fif'
digiDataFile = 'headShape_fids.fif'

# Files to write
newTransFifName = 'head_mri-trans.fif'

subjects_dir = "../subjects"
MRsubject = "fsaverage"

info = read_info(os.path.join(dataDir, 'proc_data', subjectID, fifName))

plot_kwargs = dict(
    subject=MRsubject,
    subjects_dir=subjects_dir,
    surfaces="head",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

# Add digitization to the MEG info
digi = mne.channels.read_dig_fif(os.path.join(dataDir, 'proc_data', subjectID, 'digi', digiDataFile))
info.set_montage(digi)

# Set up initial registration (no rotation or translation applied)
fiducials = "auto"  # get fiducials I defined from the bem folder
coreg = Coregistration(info, MRsubject, subjects_dir, fiducials=fiducials)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

# Fit to the fiducials (MEG and MRI)
coreg.fit_fiducials(verbose=True)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

# Initial fit to the headshape using ICP
coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
#fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

# Drop any points that are far from the head surface
coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

# Refit to the ICP with reduced point set
coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(
    f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
    f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
)

# Manual tweak of position if needed
manualTrans = [-0.005,-0.005,0.007]  # manual adjustment (translation) in metres


manTransMatrix = copy.deepcopy(coreg).trans.get('trans')
manTransMatrix[0:3,3] = manTransMatrix[0:3,3] + manualTrans
manTrans = mne.transforms.Transform(fro="head", to="mri", trans=manTransMatrix)

# Plot the final registration
fig = mne.viz.plot_alignment(info, trans=manTrans, **plot_kwargs)
mne.viz.set_3d_view(fig, **view_kwargs)

mne.write_trans(os.path.join(dataDir, 'proc_data', subjectID, newTransFifName), manTrans, overwrite=True)

