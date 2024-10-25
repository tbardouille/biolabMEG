import open3d as o3d
import os
import numpy as np
import copy
import mne

subjectID = 'mnsbp001'

# Files to read
dataDir = '../'
fidPly = os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'fiducials.ply')
headPly = os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'headScan.ply')

# Files to write
rotatedFidPly = os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'fiducials_headCoords.ply')
transHeadPly = os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'headScan_headCoords.ply')
transDecimHeadPly = os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'headScan_headCoords_downSampled.ply')

# Read in files
fiducials = o3d.io.read_point_cloud(fidPly)
head = o3d.io.read_point_cloud(headPly)

# Find clusters of points in the fiducials data. There should be three
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        fiducials.cluster_dbscan(eps=2, min_points=10, print_progress=True))

# Push on if there are three clusters
if labels.max() == 2:
    
    CofMs = []
    # Find the centre of mass of each cluster
    for fidNum in range(3):
        
        # Find which points relate to this cluster
        fidIndex = np.where(labels==fidNum)[0]
        # Get the points that relate to this cluster 
        fidPts = np.asarray([np.asarray(fiducials.points)[x,:] for x in fidIndex])
        # Get the centre of mass of those points 
        fidCofM = np.mean(fidPts, axis=0)
        
        # Compile centre of masses across loops
        CofMs.append( fidCofM )
CofMs = np.asarray(CofMs)

# Nasion has the smallest z value
nasInd = np.argmin(CofMs[:,2])
NAS = CofMs[nasInd,:]

# Left PA has the largest y value
leftInd = np.argmax(CofMs[:,1])
LPA = CofMs[leftInd,:]

# Right PA has the smallest y value
rightInd = np.argmin(CofMs[:,1])
RPA = CofMs[rightInd,:]

# Reorder the points
fiducialPts = np.vstack((NAS, np.vstack((LPA, RPA))))

# Define the head coordinate frame
Xslope = RPA-LPA
xhat = Xslope/np.sqrt(np.dot(Xslope,Xslope))
t_numer = xhat[0]*(NAS[0]-LPA[0]) + xhat[1]*(NAS[1]-LPA[1]) + xhat[2]*(NAS[2]-LPA[2]) 
t_denom= np.dot(xhat,xhat)
t = t_numer/t_denom
origin = LPA + t*xhat
Yslope = NAS-origin
yhat = Yslope/np.sqrt(np.dot(Yslope,Yslope))
zhat = np.cross(xhat,yhat)

# Define the rotation matrix for the new coordinate frame
rotMatrix = np.hstack((xhat,0))
rotMatrix = np.vstack((rotMatrix, np.hstack((yhat,0))))
rotMatrix = np.vstack((rotMatrix, np.hstack((zhat,0))))
rotMatrix = np.vstack((rotMatrix, np.asarray([0,0,0,1])))

# Translate the fiducials to the new origin
fid_translate = copy.deepcopy(fiducials).translate(-origin)

# Rotate and save the translated fiducials
fiducials_headCoords = copy.deepcopy(fid_translate).transform(rotMatrix)
o3d.io.write_point_cloud(rotatedFidPly, fiducials_headCoords)

# Translate, rotate and save the head scan in the head coordinates
head_headCoords = copy.deepcopy(head).translate(-origin).transform(rotMatrix)
o3d.io.write_point_cloud(transHeadPly, head_headCoords)

# Find clusters of points in the fiducials in head coords
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        fiducials_headCoords.cluster_dbscan(eps=2, min_points=10, print_progress=True))
if labels.max() == 2:
    CofMs = []
    # Find the centre of mass of each cluster
    for fidNum in range(3):
        # Find which points relate to this cluster
        fidIndex = np.where(labels==fidNum)[0]
        # Get the points that relate to this cluster 
        fidPts = np.asarray([np.asarray(fiducials_headCoords.points)[x,:] for x in fidIndex])
        # Get the centre of mass of those points 
        fidCofM = np.mean(fidPts, axis=0)
        # Compile centre of masses across loops
        CofMs.append( fidCofM )
CofMs = np.asarray(CofMs)*1e-3
# Nasion has the largest y value
nasInd = np.argmax(CofMs[:,1])
NAS = CofMs[nasInd,:]
# Left PA has the smallest x value
leftInd = np.argmin(CofMs[:,0])
LPA = CofMs[leftInd,:]
# Right PA has the largest xvalue
rightInd = np.argmax(CofMs[:,0])
RPA = CofMs[rightInd,:]

# Make a heavily downsampled (decimated) head for MEG-MRI registration
headDecim = copy.deepcopy(head_headCoords).voxel_down_sample(voxel_size=10)
headShape = np.asarray(headDecim.points)*1e-3   # Now in metres
# Keep only points with z > -0.03
a = np.where(headShape[:,2]>-0.03)[0]
headShape = headShape[a,:]
# Keep only points with z < 0.07
a = np.where(headShape[:,2]<0.07)[0]
headShape = headShape[a,:]
# Keep only points with y > 0.03m
a = np.where(headShape[:,1]>0.03)[0]
headShape = headShape[a,:]

# Save as a fif with fids and head points for opening in mne coreg
headMontage = mne.channels.make_dig_montage(nasion=NAS, lpa=LPA, rpa=RPA,coord_frame='head', hsp=headShape)
headMontage.save(os.path.join(dataDir, 'proc_data', subjectID, 'digi', 'headShape_fids.fif'), overwrite=True)
