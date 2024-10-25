import mne
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import os
import copy
import scipy.signal as ss
import scipy.stats as sstats
from sklearn import linear_model
import matplotlib.pyplot as plt

from mne.evoked import combine_evoked
from mne.forward import make_forward_dipole
from mne.simulation import simulate_evoked
from mne.utils import _verbose_safe_false

###
# Phantom data analysis for scans with 13 sensors and 3 references
#	from June 26, updated to not have to turn off unused channels for H. See HACK
#	Using new helmet
#	Now with 4 dipoles and 4 HPI coils
#   Updated thetas: 3rd dipole was 15, now 35


#####################
### THIS CODE IS HACKED!!!! 
#
# See line 483 where we drop unused channels from phantomH because 
#	H only uses 3 channels
#
######################
# Scans recorded 

runs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# Date of scan
date = '20241018'
# File prefix
fifPre = 'phantom'
# Make plots?
plotOK = False
# Write data?
writeOK = True
# Other (fixed) variables
dataDir = '../raw_data'
outputDir = '../results/' + date
supportDir = './supportFiles'
digName = 'Default Sensor locations - AllSensors_48mm_fixed.xls' 

shcOrder=1
windowDuration = 4  # seconds
n_fft = 2000

# ECD Locations
rho = 6.5
phis = [0, 0, 180, 180] # degrees
thetas = [65, 25, 35, 55] # degrees
numActivations = 100

# HPI Locations
#	Going from +x axis towards +y, HPIs are at th 45 degrees in the order
#	 
hpiVal = 7/np.sqrt(2)	# HPI positions in cm
hpiLocs = [	[-hpiVal, -hpiVal, 0],
			[-hpiVal, +hpiVal, 0],
			[hpiVal, hpiVal, 0],
			[hpiVal, -hpiVal, 0]
		]

# Other source variables used in the analysis
numSources = 8
numECDs = 4
numHPIs = 4

epochStart = -0.200
epochEnd = 0.200
baselineStart = -0.150
baselineEnd = -0.100
HPIFitSample = 200 		# sample in the evoked data for HPI (mag dip) fitting
						#	should be in the middle of the epoch
ECDFitSample = 5 		# sample in the evoked data for ECD fitting

l_freq=1
h_freq=20

# New subsets for Helmet v2 (missing 10 slots as compared to v1 helmet)
channelSubsets = [ 	['Fpz', 'AFpz', 'AFz', 						# A
						'AFp3', 'Fp1', 'AF1', 'AF3', 'AF5', 
						'AFp4', 'Fp2', 'AF2', 'AF4', 'AF6'],
					['Fz', 										# B
						'AF7', 'F1', 'F3', 'F5', 'F7', 'F9',
						'AF8', 'F2', 'F4', 'F6', 'F8', 'F10'],
					['FCz', 									# C
						'FC1', 'FC3', 'FC5', 'FC7', 'FC9', 'FT7',
						'FC2', 'FC4', 'FC6', 'FC8', 'FC10', 'FT8'],
					['CPz', 									# D
						'C1',  'C3', 'C5', 'C7', 'CP1', 'CP3', 
						'C2', 'C4', 'C6', 'C8', 'CP2', 'CP4'],	
					['Pz', 										# E 
						'CP5', 'CP7', 'CP9', 'TP7', 'P1', 'P3', 
						'CP6', 'CP8', 'CP10', 'TP8', 'P2', 'P4'], 
					['POz', 									# F 
						'P5', 'P7', 'P9', 'P11', 'PO1', 'PO3', 
						'P6', 'P8', 'P10', 'P12', 'PO2', 'PO4'],
					['Oz', 										# G 
						'PO5', 'PO7', 'PO9', 'O1', 'O5', 'O9', 
						'PO6', 'PO8', 'PO10', 'O2', 'O6', 'O10'],	
					['OIz', 'OIhz', 'Iz',]						# H 
				]

################################################
# Functions

def getSensorLocations(supportDir, digName):
	# Get file with default sensor locations for this sensor layout
	digDf = pd.read_excel(os.path.join(supportDir, digName))

	# Make the array to set sensor locations
	locColumns = ['sensor_x', 'sensor_y', 'sensor_z', 'ex_i',
	       'ex_j', 'ex_k', 'ey_i', 'ey_j', 'ey_k', 'ez_i', 'ez_j', 'ez_k']
	locArray = np.asarray(digDf.loc[:,locColumns])

	sensorNames = digDf['Sensor Names'].to_list()

	return locArray, sensorNames

def updateSensorNamesAndLocations(raw, thisRun, channelSubset, locArray, sensorNames):

	# Determine the relevant indices for channels of interest
	channelSubsetIndex = [sensorNames.index(i) for i in channelSubset]
	numChannels = len(channelSubsetIndex)

	# Take only relevant locations
	locSubArray = locArray[channelSubsetIndex,:]

	# Replace locations in raw.info
	newRaw = raw.copy()
	offsetIndex = 0
	for c in np.arange(numChannels):
	    channelIndex = c+offsetIndex
	    newRaw.info['chs'][channelIndex]['loc'] = locSubArray[c,:]

	# Rename channels based on the dataset (to avoid duplicate sensor names in the combined evoked)
	numSensors = len(channelSubset)
	channelNames = newRaw.info['ch_names'][offsetIndex:numSensors]
	ctr = 0
	nameDict = {}
	for ch in channelNames:
		nameDict[ch] = channelSubset[ctr]
		ctr += 1
	newRaw.rename_channels(nameDict)

	return newRaw

def referenceArrayRegression(raw_filter, opmChannels, sensorChannels, refChannels):

	# Window data (1 second cosine) to clean out high-pass edge effects
	opmData = raw_filter.get_data()[opmChannels]

	# Remove signals related to reference signals via regression
	sensorData = opmData[sensorChannels,:]
	referenceData = opmData[refChannels,:]

	numSensors = len(sensorChannels)
	regressData = copy.copy(sensorData)
	for i in np.arange(numSensors):
		# Put data into a pandas dataframe
		data = {'sensor': sensorData[i,:],
				'Xref': referenceData[0,:],
				'Yref': referenceData[1,:],
				'Zref': referenceData[2,:],
				}
		df = pd.DataFrame(data)
		x = df[['Xref','Yref', 'Zref']]
		y = df['sensor']
		# Run multi-variable regression
		regr = linear_model.LinearRegression()
		regr.fit(x, y)
		# Extract cleaned sensor data 
		regressData[i,:] = sensorData[i,:] - regr.coef_[0]*referenceData[0,:] - regr.coef_[1]*referenceData[1,:] - regr.coef_[2]*referenceData[2,:]

	# Put cleaned data into a raw_regress object
	allData = raw_filter.get_data()
	allData[sensorChannels,:] = regressData
	raw_regressed = mne.io.RawArray(allData, raw_filter.info)

	return raw_regressed

def detrendData(raw, opmChannels):

	# Linear regression to baseline correct and detrend
	opmData = raw.copy().pick('mag').get_data()
	opmDataBaseCorr = copy.copy(opmData)
	for i in np.arange(opmData.shape[0]):
		result = sstats.linregress(raw.times,opmData[i,:])
		modelFit = result.intercept + result.slope*raw.times
		opmDataBaseCorr[i,:] = opmData[i,:] - modelFit

	# Put detrended data into a raw object
	allData = raw.get_data()
	allData[opmChannels,:] = opmDataBaseCorr
	raw_detrend = mne.io.RawArray(allData, raw.info)

	return raw_detrend

def windowData(raw, opmChannels, winDur):

	# Window data (1 second cosine) to clean out high-pass edge effects
	opmData = raw.copy().pick('mag').get_data()
	numSamples = opmData.shape[1]
	numChannels = opmData.shape[0]
	fs = raw.info['sfreq']
	cosTimes =  np.arange(winDur*fs)/fs
	cosWin = np.expand_dims(-0.5*np.cos(2*np.pi*0.5*cosTimes/winDur)+0.5,0)	# half a cosine
	middleWin = np.ones((1,numSamples-2*len(cosTimes)))
	fullWindow = np.tile(np.hstack((cosWin, middleWin, cosWin[:,::-1])), (numChannels,1))
	windowedData = opmData*fullWindow

	# Put detrended data into a raw object
	allData = raw.get_data()
	allData[opmChannels,:] = windowedData
	raw_win = mne.io.RawArray(allData, raw.info)

	return raw_win

def getEvents(raw, voltageThreshold):

	# Get the stim channel, which has different voltage values for each interval type in the experiment
	stimChan = mne.channel_indices_by_type(raw.info)['stim'][0]
	stimData = raw.get_data()[stimChan,:]
		
	# Using the phantom driver signal to get events
	peakIndex = ss.find_peaks(stimData, height=voltageThreshold, distance=100)[0]
	numTrials = len(peakIndex)
	print('There are ' + str(numTrials) + ' trials.')

	# Make a list of the onset time of each stimulus
	events = np.vstack((peakIndex, np.zeros(numTrials)))
	events = np.vstack((events, np.ones(numTrials))).T.astype(int)

	return events	

def calcEvoked(raw, events, project, numSources, numActivations, epochStart, epochEnd, baselineStart, baselineEnd):

	# Split the raw data into epochs per source
	# 	Epoch and average 
	evokeds = []
	for s in np.arange(numSources):
		firstEvent = s*numActivations
		lastEvent = (s+1)*numActivations
		epochs = mne.Epochs(raw, events[firstEvent:lastEvent,:], tmin=epochStart, tmax=epochEnd, 
			baseline=(baselineStart,baselineEnd), preload=True, proj=project)
		evoked = epochs.average().apply_baseline((baselineStart,baselineEnd))
		# Drop the last three channels (reference array)
		evoked.drop_channels(evoked.info['ch_names'][-3:])
		evokeds.append(evoked)

	return evokeds

def flip_sensors(raw, sens1Ind, sens2Ind):

	data = raw.get_data()
	newData = copy.copy(data)
	newData[sens1Ind,:] = data[sens2Ind,:]
	newData[sens2Ind,:] = data[sens1Ind,:]
	newRaw = mne.io.RawArray(newData, raw.info)

	return newRaw

def combineProjection(combinedEvoked, startInd, stopInd):

	allNames = []
	allData = [] 
	for p in np.arange(startInd, stopInd):
		thisName = combinedEvoked.info['projs'][p]['data']['col_names']
		allNames.append(thisName)
		thisData = combinedEvoked.info['projs'][p]['data']['data'][0]
		allData.append(thisData)
	col_names = [x for xs in allNames for x in xs]
	data = [x for xs in allData for x in xs]
	ncol = len(data)
	newProj1={}
	newProj1['data']={}
	newProj1['data']['col_names'] = col_names
	newProj1['data']['row_names'] = None
	newProj1['data']['data'] = np.asarray([data])
	newProj1['data']['ncol'] = ncol
	newProj1['data']['nrow'] = 1
	newProj1['desc'] = 'HFC: l=1 m=0'
	newProj1['kind'] = 1
	newProj1['active'] = False
	proj1 = mne.Projection(data=newProj1['data'], desc=newProj1['desc'], kind=newProj1['kind'], active=newProj1['active'], explained_var=None)

	return proj1

def combineEvokedDatasets(allEvokeds):
	
	# Combine all channels into one list of evoked variable (one evoked per source_)
	combinedEvokeds = allEvokeds[0].copy()
	for evs in allEvokeds[1::]:
	    for s in np.arange(numSources):
	        combinedEvokeds[s].add_channels([evs[s]])

	# Combine HFC projectors
	if len(allEvokeds[0][0].info['projs'])>0:
		numRuns = len(allEvokeds)
		proj1 = combineProjection(combinedEvokeds[0], 0, numRuns)
		proj2 = combineProjection(combinedEvokeds[0], numRuns, numRuns*2)
		proj3 = combineProjection(combinedEvokeds[0], numRuns*2, numRuns*3)

	# Write new projectors to the evoked data
	newCombinedEvokeds = []
	for ev in combinedEvokeds:
		if len(allEvokeds[0][0].info['projs'])>0:
			newEv = ev.copy().add_proj([proj1, proj2, proj3], remove_existing=True, verbose=False).apply_proj(verbose="error")
		else:
			newEv = ev.copy()
		newCombinedEvokeds.append(newEv)

	return newCombinedEvokeds

def plot_raw_data(raw1, raw2, raw3, raw4):
	fig, ax = plt.subplots(2,4)

	fmax = 50
	yRange = [10,1e4]
	thisRaw = raw1.copy()
	i=0
	ax[0,i].plot(thisRaw.times, thisRaw.copy().pick('mag').get_data().T)
	ax[0,i].set_title('BandPass + Notch')
	PSD = thisRaw.compute_psd(n_fft=2000, exclude='bads')
	PSD.plot(axes=ax[1,i], dB=False, show=False, amplitude=True)
	ax[1,i].set_xlim([0,fmax])
	ax[1,i].set_yscale('log')
	ax[1,i].set_ylim(yRange)

	thisRaw = raw2.copy()
	i=1
	ax[0,i].plot(thisRaw.times, thisRaw.copy().pick('mag').get_data().T)
	ax[0,i].set_title('Detrended')
	PSD = thisRaw.compute_psd(n_fft=2000, exclude='bads')
	PSD.plot(axes=ax[1,i], dB=False, show=False, amplitude=True)
	ax[1,i].set_xlim([0,fmax])
	ax[1,i].set_yscale('log')
	ax[1,i].set_ylim(yRange)

	thisRaw = raw3.copy()
	i=2
	ax[0,i].plot(thisRaw.times, thisRaw.copy().pick('mag').get_data().T)
	ax[0,i].set_title('HFC')
	PSD = thisRaw.compute_psd(n_fft=2000, exclude='bads')
	PSD.plot(axes=ax[1,i], dB=False, show=False, amplitude=True)
	ax[1,i].set_xlim([0,fmax])
	ax[1,i].set_yscale('log')
	ax[1,i].set_ylim(yRange)

	thisRaw = raw4.copy()
	i=3
	ax[0,i].plot(thisRaw.times, thisRaw.copy().pick('mag').get_data().T)
	ax[0,i].set_title('Regressed')
	PSD = thisRaw.compute_psd(n_fft=2000, exclude='bads')
	PSD.plot(axes=ax[1,i], dB=False, show=False, amplitude=True)
	ax[1,i].set_xlim([0,fmax])
	ax[1,i].set_yscale('log')
	ax[1,i].set_ylim(yRange)

	plt.show()

def dipoleFitEvoked(supportDir, combinedEvoked):

	################
	# Set up source inside a sphere
	bemSphere = mne.make_sphere_model(r0=(0,0,0.0), head_radius=0.08, info=combinedEvoked.info)

	cov = mne.make_ad_hoc_cov(combinedEvoked.info)

	# Identity transformation between head and mri space
	trans2 = np.eye(4)
	trans2 = mne.transforms.Transform('head', 'mri', trans=trans2) 

	# Do the dipole fitting
	dip_opm, _ = mne.fit_dipole(combinedEvoked.copy().crop(-0.005,0.005), cov, bemSphere, trans2, verbose=False)

	# Extract the position at the peak of the 5 Hz signal (50 ms) in cm
	pos = dip_opm.pos[5,:]*100

	return pos, dip_opm

def magDipoleFit(B, info):

	x0 = np.asarray([.065, .065, .02])	# metres ?
	too_close = 'raise'
	safe_false = _verbose_safe_false()

	meg_coils = mne.chpi._concatenate_coils(mne.chpi._create_meg_coils(info["chs"], "accurate"))

	cov = mne.cov.make_ad_hoc_cov(info, verbose=safe_false)
	whitener, _ = mne.cov.compute_whitener(cov, info, verbose=safe_false)

	R = np.linalg.norm(meg_coils[0], axis=1).min()
	guesses = mne.chpi._make_guesses(
	    dict(R=R, r0=np.zeros(3)), 0.01, 0.0, 0.005, verbose=safe_false
	)[0]["rr"]

	fwd = mne.chpi._magnetic_dipole_field_vec(guesses, meg_coils, too_close)
	fwd = np.dot(fwd, whitener.T)
	fwd.shape = (guesses.shape[0], 3, -1)
	fwd = np.linalg.svd(fwd, full_matrices=False)[2]
	guesses = dict(rr=guesses, whitened_fwd_svd=fwd)

	x, gof, moment = mne.chpi._fit_magnetic_dipole(B, x0, too_close, whitener, meg_coils, guesses)

	return x, gof, moment 

def plot_topos(evoked, pred_evoked, diff_evoked, dsNum, dipNum, colourRange):

	# remember to create a subplot for the colorbar
	fig, axes = plt.subplots(
	    nrows=1,
	    ncols=4,
	    figsize=[10.0, 3.4],
	    gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1], top=0.85),
	    layout="constrained",
	)
	plot_params = dict(times=[0], ch_type="mag", outlines="head", colorbar=False, vlim=(-colourRange,colourRange))

	# first plot the topography at the time of the best fitting (single) dipole
	evoked.plot_topomap(time_format="Measured field", axes=axes[0], **plot_params)

	# then plot the predicted field
	pred_evoked.plot_topomap(time_format="Predicted field", axes=axes[1], **plot_params)

	# Now add in subtraction plot
	plot_params["colorbar"] = True
	diff_evoked.plot_topomap(time_format="Difference", axes=axes[2:], **plot_params)
	fig.suptitle(
	    "Dipole Fit "
	    "for source {:d} in dataset {:d}".format(dipNum, dsNum),
	    fontsize=16,
	)
	return fig

def xyz2rpt(xyz):
    '''
    in: xyz - [x,y,z]
    out: rpt - [r,phi,theta]
    '''
    x,y,z = xyz
    r = np.sqrt(x**2+y**2+z**2)
    rxy = np.sqrt(x**2+y**2)
    theta = np.arctan(rxy/z)*180/np.pi
    phi = np.arctan2(y,x)*180/np.pi
    return np.array([r,phi,theta])

def rpt2xyz(rpt):
    '''
    in: rpt - [r,phi,theta]
    out: xyz - [x,y,z]
    '''
    r, phi, theta = rpt
    phi = phi*np.pi/180
    theta = theta*np.pi/180
   
    x=r*np.sin(theta)*np.cos(phi)
    y=r*np.sin(theta)*np.sin(phi)
    z=r*np.cos(theta)
    return np.array([x,y,z])

def ampSpec(raw, n_fft):
	# Amplitude spectra 
	PSDs = raw.compute_psd(n_fft=n_fft, exclude='bads')
	ampSpecData = np.sqrt( PSDs.get_data() ) * 1e15 # in fT
	freq = PSDs.freqs

	return ampSpecData, freq

################################################
# Main Program

# Loop over datasets with subsets of sensors on the helmet
allEvokeds_det = []
allEvokeds_hfc = []
allEvokeds_regr = []
ampSpec_filts = []
ampSpec_hfcs = []
for r in np.arange(len(runs)):

	thisRun = runs[r]
	thisSubset = channelSubsets[r]

	# Read sensor info used in the acquisiion (default depth)
	#	and place positions in an array
	locArray, sensorNames = getSensorLocations(supportDir, digName)

	# Read the raw data and calculate the evoked response
	rawFif = fifPre + str(thisRun) + '_raw.fif'
	rawFile = os.path.join(dataDir, date, rawFif)

	# Load the data
	raw = mne.io.read_raw_fif(rawFile, preload=True)
	'''

	# HACK! Drop unneeded channels from phantomH
	if r==7:
		raw.drop_channels(['FL0104-BZ_CL', 
			'FL0105-BZ_CL', 
			'FL0106-BZ_CL', 
			'FL0107-BZ_CL', 
			'FL0108-BZ_CL', 
			'FL0109-BZ_CL', 
			'FL0110-BZ_CL', 
			'FL0111-BZ_CL', 
			'FL0112-BZ_CL', 
			'FL0113-BZ_CL'])
	'''
	raw_meg = raw.copy().pick('mag')

		
	# Extract phantom peak activations timing
	events = getEvents(raw, 2)

	# Get indices for sensors and references
	opmIndices = mne.channel_indices_by_type(raw_meg.info)['mag']
	numChannels = len(opmIndices)
	numSensors = numChannels-3
	sensorIndices = opmIndices[0:numSensors]
	referenceIndices = opmIndices[-3::]

	# Update sensor names and locations
	raw_loc = updateSensorNamesAndLocations(raw_meg.copy(), thisRun, thisSubset, locArray, sensorNames)


	# Set the last three sensors as "bad" so they are:
	#	automatically excluded from HFC
	#	manually excluded from PSD
	#	included (but irrelevant) for reference array regression
	refChannels = raw_loc.info['ch_names'][-3::]
	raw_loc.info['bads'].extend(refChannels)

	# Filter and downsample
	raw_filt = raw_loc.copy()
	raw_filt = raw_filt.notch_filter([60, 120, 180])
	raw_filt = raw_filt.filter(l_freq=l_freq, h_freq=h_freq)

	# DC offset and detrend
	raw_detrend = windowData(detrendData(raw_filt.copy(), opmIndices), opmIndices, windowDuration)

	# Grab the amplitude spectrum
	ampSpec_filt, freq = ampSpec(raw_detrend.copy(), n_fft)
	
	# Reference array regression 
	raw_regressed = referenceArrayRegression(raw_detrend.copy(), opmIndices, sensorIndices, referenceIndices)

	# Homogenous field compensation (HFC)
	raw_hfc = raw_regressed.copy()
	projs = mne.preprocessing.compute_proj_hfc(raw_hfc.info, order=shcOrder)
	raw_hfc.add_proj(projs)
	raw_hfcApplied = raw_hfc.copy().apply_proj(verbose="error")

	# Grab the amplitude spectrum
	ampSpec_hfc, freq = ampSpec(raw_hfcApplied, n_fft)
	
	# Calculate the evoked responses (HFC projection is carried thru but not necessarily applied)
	project = False
	evokeds_hfc = calcEvoked(raw_hfc, events, project, numSources, numActivations, epochStart, epochEnd, baselineStart, baselineEnd)
	evokeds_regr = calcEvoked(raw_regressed, events, project, numSources, numActivations, epochStart, epochEnd, baselineStart, baselineEnd)
	evokeds_det = calcEvoked(raw_detrend, events, project, numSources, numActivations, epochStart, epochEnd, baselineStart, baselineEnd)

	ampSpec_filts.append ( ampSpec_filt )
	ampSpec_hfcs.append ( ampSpec_hfc )
	allEvokeds_hfc.append( evokeds_hfc )
	allEvokeds_regr.append( evokeds_regr )
	allEvokeds_det.append( evokeds_det )

# Stack all amplitude spectra into one array
for r in np.arange(len(runs)):
	if r==0:
		ampSpec_filt = copy.copy(ampSpec_filts[r])
		ampSpec_hfc = copy.copy(ampSpec_hfcs[r])
	else:
		ampSpec_filt = np.vstack((ampSpec_filt, ampSpec_filts[r]))
		ampSpec_hfc = np.vstack((ampSpec_hfc, ampSpec_hfcs[r]))
ampSpecData = np.asarray( [ampSpec_filt, ampSpec_hfc])
ampSpec_mean = np.mean(ampSpecData, axis=1)

# Shielding Factor
shieldFactorSpectrum = 20*np.log10(ampSpecData[0,:,:]/ampSpecData[1,:,:])
shieldFactor_mean = np.mean(shieldFactorSpectrum, axis=0)

# Combine consecutive evokeds datasets
combinedEvokeds_hfc = combineEvokedDatasets(allEvokeds_hfc)
combinedEvokeds_regr = combineEvokedDatasets(allEvokeds_regr)
combinedEvokeds_det = combineEvokedDatasets(allEvokeds_det)

# Save averaged MEG data
if writeOK:
	for s in np.arange(numSources):
		outFif = os.path.join(outputDir, fifPre + 'source' + str(s+1) + '-ave.fif')
		combinedEvokeds_hfc[s].save(outFif, overwrite=True)

# Fit one current dipole for each ECD source 
ECDFitData = []
for s in np.arange(numECDs):

	fitData = {}

	# Fit dipole for one source
	pos, dip_opm = dipoleFitEvoked(supportDir, combinedEvokeds_hfc[s])
	
	# Calculate error - Cartesian
	fitData['knownPosition_rpt'] = [rho, phis[s], thetas[s]] 
	fitData['knownPosition_xyz'] = rpt2xyz(fitData['knownPosition_rpt'])
	fitData['deltaR'] = pos - fitData['knownPosition_xyz']

	# Calculate error - Polar
	fitData['rpt'] = xyz2rpt(pos)
	delta_rpt = fitData['rpt'] - fitData['knownPosition_rpt']

	# Calculate localization error
	fitData['LEmag'] = np.sqrt(np.sum(fitData['deltaR']*fitData['deltaR']))

	fitData['pos'] = pos
	fitData['dip'] = dip_opm

	ECDFitData.append(fitData)

# Fit one magnetic dipole for each HPI source 
HPIFitData = []
ctr = 0
for s in np.arange(numHPIs)+numECDs:

	# Fit magnetic dipole (in device coordinate frame)
	B_hpi = combinedEvokeds_hfc[s].data[:,HPIFitSample].T
	x, gof, moment = magDipoleFit(B_hpi, combinedEvokeds_hfc[s].info)
	
	fitData = {}
	fitData['pos'] = x*100			# cm
	fitData['gof'] = gof*100		# %
	fitData['amplitude'] = moment

	fitData['knownPosition_xyz'] = hpiLocs[ctr]
	fitData['knownPosition_rpt'] = xyz2rpt(hpiLocs[ctr])
	fitData['deltaR'] = fitData['pos'] - fitData['knownPosition_xyz']
	fitData['LEmag'] = np.sqrt(np.sum(fitData['deltaR']*fitData['deltaR']))
	fitData['rpt'] = xyz2rpt(pos)

	HPIFitData.append(fitData)

	ctr = ctr + 1

# Plot all localized points to one graph
if plotOK:
	colSet = ['b','c','r', 'g']
	mrkSet = ['*', '+', 'o', '^']
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	for ds in np.arange(numECDs):
		a = ECDFitData[ds]['pos']
		ax.scatter(a[0], a[1], a[2], c=colSet[ds])
	for ds in np.arange(numHPIs):
		a = HPIFitData[ds]['pos']
		ax.scatter(a[0], a[1], a[2], c='k', marker=mrkSet[ds])
	ax.set_xlim((-7,7))
	ax.set_ylim((-7,7))
	ax.set_zlim((-1,7))
	ax.set_xlabel('x [cm]')
	ax.set_ylabel('y [cm]')
	ax.set_zlabel('z [cm]')
	plt.show()


# Output results to files (csv and dip)
if writeOK:

	resultsFile = os.path.join(outputDir, fifPre + 'Results.csv')
	with open(resultsFile, 'w') as f:

		# Column headers
		f.writelines("Source\tRho_Known\tPhi_Known\tTheta_Known\tx_Known\ty_Known\tz_Known\t")
		f.writelines("Rho_Pred\tPhi_Pred\tTheta_Pred\tx_Pred\ty_Pred\tz_Pred\t")
		f.writelines("Qmag\t")
		f.writelines("x_Delta\ty_Delta\tz_Delta\tLE\n")

		for s in np.arange(numECDs):

			# Save the dipole data
			dipFile = os.path.join(outputDir, fifPre + 'source' + str(s+1) + '-ave.dip')
			dip_opm = ECDFitData[s]['dip']
			dip_opm.save(dipFile, overwrite=True)

			knownPosition_rpt = ECDFitData[s]['knownPosition_rpt']
			knownPosition_xyz = ECDFitData[s]['knownPosition_xyz']
			rpt = ECDFitData[s]['rpt']
			pos = ECDFitData[s]['pos']
			Qmag = ECDFitData[s]['dip'][ECDFitSample].amplitude[0]*1e9
			deltaR = ECDFitData[s]['deltaR']
			LEmag = ECDFitData[s]['LEmag']
			f.writelines("Q{:d}\t".format(s+1))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(knownPosition_rpt[0], knownPosition_rpt[1], knownPosition_rpt[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(knownPosition_xyz[0], knownPosition_xyz[1], knownPosition_xyz[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(rpt[0], rpt[1], rpt[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t".format(pos[0], pos[1], pos[2], Qmag))
			f.writelines("{:.2f}\t{:.2f}\t{:.2f}\t".format(deltaR[0], deltaR[1], deltaR[2]))
			f.writelines("{:.3f}\n".format(LEmag))

		for s in np.arange(numHPIs):

			knownPosition_rpt = HPIFitData[s]['knownPosition_rpt']
			knownPosition_xyz = HPIFitData[s]['knownPosition_xyz']
			rpt = HPIFitData[s]['rpt']
			pos = HPIFitData[s]['pos']
			Q = HPIFitData[s]['amplitude']
			Qmag = np.sqrt(np.sum(Q*Q))*1e9
			deltaR = HPIFitData[s]['deltaR']
			LEmag = HPIFitData[s]['LEmag']
			f.writelines("HPI{:d}\t".format(s+1))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(knownPosition_rpt[0], knownPosition_rpt[1], knownPosition_rpt[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(knownPosition_xyz[0], knownPosition_xyz[1], knownPosition_xyz[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t".format(rpt[0], rpt[1], rpt[2]))
			f.writelines("{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t".format(pos[0], pos[1], pos[2], Qmag))
			f.writelines("{:.2f}\t{:.2f}\t{:.2f}\t".format(deltaR[0], deltaR[1], deltaR[2]))
			f.writelines("{:.3f}\n".format(LEmag))

	f.close()
	
'''
# Project evoked data to the dipole location
## NOTE: I INCREASED THE HEAD_RADIUS TO MAKE THIS WORK
trans2 = np.eye(4)
trans2 = mne.transforms.Transform('head', 'mri', trans=trans2) 
bemSphere = mne.make_sphere_model(r0=(0,0,0.0), head_radius=0.10, info=combinedEvokeds_hfc[0].info)
cov = mne.make_ad_hoc_cov(combinedEvokeds_hfc[0].info)
dipData = [] 
for s in np.arange(numSources):
	bestFit = allFitData[s]['dip'][5]
	sourceData1, residualEvoked = mne.fit_dipole(combinedEvokeds_det[s], cov, bemSphere, trans2, pos=bestFit.pos[0], ori=bestFit.ori[0])
	thisDip = sourceData1.data[0,:]
	sourceData2, residualEvoked = mne.fit_dipole(combinedEvokeds_hfc[s], cov, bemSphere, trans2, pos=bestFit.pos[0], ori=bestFit.ori[0])
	thatDip = sourceData2.data[0,:]
	sourceData3, residualEvoked = mne.fit_dipole(combinedEvokeds_regr[s], cov, bemSphere, trans2, pos=bestFit.pos[0], ori=bestFit.ori[0])
	thotherDip = sourceData3.data[0,:]
	dipData.append([thisDip, thatDip, thotherDip])
dipData = np.asarray(dipData)*1e9

#######################################
# Plot dipole time courses for evoked in two noise reduction methods
if plotOK:
	fig, axs = plt.subplots(1, 3, sharey=True)
	axs_flat = np.ndarray.flatten(axs)

	axs_flat[0].plot(sourceData1.times, np.squeeze(dipData[:,1,:]).T)
	axs_flat[0].set_ylabel('Dipole Strength [nAm]')
	axs_flat[0].set_xlabel('Time [s]]')
	axs_flat[0].grid(True)
	axs_flat[0].set_title('HFC')

	axs_flat[1].plot(sourceData2.times, np.squeeze(dipData[:,2,:]).T)
	axs_flat[1].grid(True)
	axs_flat[1].set_title('Regression')

	axs_flat[2].plot(sourceData2.times, np.squeeze(dipData[:,0,:]).T)
	axs_flat[2].grid(True)
	axs_flat[2].set_title('Filter + Detrend')

	plt.show()
'''
#######################################
# Plot raw data from one scan (time and amplitude spectrum)
if plotOK:
	plot_raw_data(raw_filt, raw_detrend, raw_regressed, raw_hfcApplied)

#######################################
# Plot amplitude spectra and shielding factor

if plotOK:
	fig, axs = plt.subplots(1, 3, sharex=True)
	axs_flat = np.ndarray.flatten(axs)

	axs_flat[0].loglog(freq, ampSpecData[0,:,:].T, linewidth=1)
	axs_flat[0].loglog(freq, ampSpec_mean[0,:], 'k', linewidth=3)
	axs_flat[0].set_ylabel('Field Strength [fT/root(Hz)]')
	axs_flat[0].set_ylim((1,1e4))
	axs_flat[0].set_title('Filter + Detrend')

	axs_flat[1].loglog(freq, ampSpecData[1,:,:].T, linewidth=1)
	axs_flat[1].loglog(freq, ampSpec_mean[1,:], 'k', linewidth=3)
	axs_flat[1].set_ylim((1,1e4))
	axs_flat[1].set_title('Cleaned')

	axs_flat[2].plot(freq, shieldFactorSpectrum.T, linewidth=1)
	axs_flat[2].plot(freq, shieldFactor_mean, 'k', linewidth=3)
	axs_flat[2].set_ylabel('Shielding Factor [dB]]')
	axs_flat[2].set_title('Total Shielding')

	for i in np.arange(3):
		axs_flat[i].grid(True, which='both')
		axs_flat[i].set_xlabel('Frequency [Hz]')
		axs_flat[i].set_xscale('linear')
		axs_flat[i].set_xlim((0,30))

	plt.show()

#######################################
# Evoked fields for sensor array   

if plotOK:
	fig, axs = plt.subplots(2,4)
	yscale = 1000
	colscale = (-500,500)
	for i in np.arange(4):
		combinedEvokeds_hfc[i].plot(show=False, axes=axs[0][i], ylim= dict(mag=[-yscale, yscale]), spatial_colors=False, proj=True)
		combinedEvokeds_hfc[i].plot_topomap(show=False, axes=axs[1][i], times=[0], colorbar=False, vlim=colscale, proj=True)
	fig.show()


print('**** ECDs ******')
Qs = [ECDFitData[i]['dip'][5].amplitude for i in np.arange(4)]
print(Qs)
GOFs = [ECDFitData[i]['dip'][5].gof for i in np.arange(4)]
print('Goodness of fits:')
print(GOFs)
POSs = [ECDFitData[i]['pos'] for i in np.arange(4)]
print('positions')
print(POSs)
print()


print('**** HPIs ****')
GOFs = [HPIFitData[i]['gof'] for i in np.arange(4)]
print('Goodness of fits:')
print(GOFs)
POSs = [HPIFitData[i]['pos'] for i in np.arange(4)]
print('positions')
print(POSs)

