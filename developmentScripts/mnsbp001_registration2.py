import mne
import mne.transforms as mt
import numpy as np
import os

subjectID = 'mnsbp001'

# Files to read
dataDir = '../'
fifName = 'MNS_shortISI_preprocessed-raw.fif'

# Files to write
transFifName = 'meg_head-trans.fif'
newFifName = 'MNS_shortISI_preprocessed_reg-raw.fif'

transMatrix = [
	[ 0.99935, 0.0168822, 0.0318544, 3.08522],
	[ -0.0230058, 0.978921, 0.202941, -19.3306],
	[ -0.0277568, -0.203542, 0.978673, 55.5719],
	[ 0, 0, 0, 1]
	] 
transMatrix = np.asarray(transMatrix)

# Change the translation into metres
transMatrix[0:3,3] = transMatrix[0:3,3]*1e-3

# Write the transformation to a fif file
trans = mt.Transform(fro="meg", to="head", trans=transMatrix)
trans.save(os.path.join(dataDir, 'proc_data', subjectID, transFifName), overwrite=True)

raw = mne.io.read_raw_fif(os.path.join(dataDir, 'proc_data', subjectID, fifName))
rawReg = raw.copy()
rawReg.info['dev_head_t'] = trans
rawReg.save(os.path.join(dataDir, 'proc_data', subjectID, newFifName), overwrite=True)

