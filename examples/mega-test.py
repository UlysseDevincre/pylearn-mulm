"""
Created on 23/05/2024

@author: ag271121

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel

DESCRIPTION NOT UP TO DATE
"""

import os
import glob
import shutil

try:
    from tqdm import tqdm
except:
    print("tqdm not available")

import numpy as np
import scipy
import scipy.stats as stats
import pandas as pd

import nibabel
#import brainomics.image_atlas
import nilearn
from nilearn import plotting

import mulm

import matplotlib.pyplot as plt

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script requires the matplotlib library")

import numpy as np
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask
from nilearn.image import get_data, threshold_img, smooth_img
from nilearn.mass_univariate._utils import normalize_matrix_on_axis
import nibabel as nib
import mulm
from mulm.utils import ttest_tfce

#from fsl.scripts import Text2Vest
#from fsl.utils.run import runfsl

save_ALL = False
brainomics = False
save_ALLcs = True


###################### SPECIFY THE USED DATABASE ######################
# specify the paths and names

BASE_PATH = f"/neurospin/rlink"
INPUT_CSV = os.path.join(BASE_PATH,"participants.tsv")
INPUT_FILES_DIR = os.path.join(BASE_PATH,"PUBLICATION/derivatives/cat12-vbm-v12.8.2")
OUTPUT_DIR = f"/home/ud279400/result/rlink_vbm_analysis/outputs_rlink_test"
MASK_PATH = "/neurospin/rlink/CODE/vbm_analysis/GM_brain_mask"

MODEL_NAME = 'diff_M00_M03'


########################################################################################################################
# Define color maps

oneway_cmaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
twoways_cmaps = ['seismic']

########################################################################################################################
# create the output folder if not existing
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read pop csv
pop = pd.read_csv(INPUT_CSV, sep='\t')


########################################################################################################################
# Select subset of participants
print("SELECT SUBSET OF PARTICIPANTS")

# rlink filter
# select subjects that have a session after 3 months
pop = pop[~pop["ses-M03_center"].isna()]

# check that none of them changed of site between M00 and M03
assert pop[pop["ses-M00_center"] != pop["ses-M03_center"]].shape[0] == 0

# MANUALLY EXCLUDED SUBJECTS
excluded_subs = []
pop = pop[~pop.participant_id.isin(excluded_subs)]

########################################################################################################################
# build the required maps

# sites as one-hot encoding vectors
def get_site_id(i, n_sites):
    """Get the one-hot encodeing vector of a given site. Remove the last 
    bit as it is redundant information.
    
    Arguments:
        - i: int, number of the site
        - n_sites: number of sites"""
    site_id = np.zeros(n_sites)
    site_id[i] = 1
    return site_id[:-1]


# define gender map
GENDER_MAP = {'F': 0, 'M': 1}

# no need of an age map

# build sites map
sites = pop.sort_values(by="ses-M00_center")["ses-M00_center"].unique()
n_sites = len(sites)
SITE_MAP = {sites[i]: get_site_id(i,n_sites) for i in range(n_sites)}
# print(SITE_MAP)


# add the mapped information in the participants csv and save it
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['site.num'] = pop['ses-M00_center'].map(SITE_MAP)
print(pop.shape)
print(pop[["sex", "sex.num", "age", "ses-M00_center", "site.num"]].head(10))


pop.to_csv(os.path.join(OUTPUT_DIR, "population.csv"), index=False)
n_sub = pop.shape[0]
assert n_sub == 95 - len(excluded_subs)


########################################################################################################################
# Read images
print("READ AND SMOOTH THE IMAGES")

def load_and_smooth(img_glob, fwhm=2):
    """Load and smooth (with a gaussian filter) a nifti image.
    Returns a numpy array.
    
    Arguments:
        - img_glob: path to the image (more precisely a string that 
        allows glob to find the image as the only file)
        - fwhm: full-width at half maximum of the applied gaussian 
        filter"""

    # get the image file from the regex
    img = glob.glob(img_glob)
    # check that only one file matches the filter
    assert len(img) == 1
    # Load the image
    img = nibabel.load(img[0])
    # smooth it
    img = nilearn.image.smooth_img(img, fwhm)
    # get the smoothed image's numpy array
    img = img.get_fdata()
    img = np.nan_to_num(img, nan=0.)

    return img


M00_images = []
M03_images = []
# numpy files already exist => read them directly
# for i, participant_id in enumerate(pop.participant_id):
for i, participant_id in tqdm(enumerate(pop.participant_id), total=n_sub):
    # get the M00 image
    M00_mri = load_and_smooth(os.path.join(INPUT_FILES_DIR, participant_id,"ses-M00/mri",
                                           f"mwp1ru{participant_id}_ses-M00_acq-*_run-1_T1w.nii"))
    M00_images.append(M00_mri)

    # get the M03 image
    M03_mri = load_and_smooth(os.path.join(INPUT_FILES_DIR, participant_id,"ses-M03/mri",
                                           f"mwp1ru{participant_id}_ses-M03_acq-*_run-1_T1w.nii"))
    M03_images.append(M03_mri)

    # get a ref image
    if i==0:
        #print("CHECK")
        ref_participant = participant_id
        ref_img_path = glob.glob(os.path.join(INPUT_FILES_DIR, ref_participant,"ses-M00/mri",
                                 f"mwp1ru{ref_participant}_ses-M00_acq-*_run-1_T1w.nii"))
        assert len(ref_img_path) == 1
        ref_img_path = ref_img_path[0]
        ref_img = nibabel.load(ref_img_path)


# put them in 2 4D images (1 for M00, one for M03)
M00_images = np.stack(M00_images)
M03_images = np.stack(M03_images)
M03_minus_M00 = M03_images - M00_images

assert np.count_nonzero(~np.isnan(M03_minus_M00)) == M03_minus_M00.size

print("M00 shape", M00_images.shape)
print("M03 shape", M03_images.shape)
print("M03 - M00 shape", M03_minus_M00.shape)


# save the 4D images
print("SAVE THE CONCAT IMAGES AS NUMPY ARRAYS")
np.save(os.path.join(OUTPUT_DIR, 'M00_images.npy'), M00_images)
np.save(os.path.join(OUTPUT_DIR, 'M03_images.npy'), M03_images)
np.save(os.path.join(OUTPUT_DIR, 'M03_minus_M00.npy'), M03_minus_M00)


########################################################################################################################
# Apply mask to keep only grey matter
print("Apply mask to keep only grey matter")

# load the mask
mask_arr = np.load(os.path.join(MASK_PATH, "binary_brain_mask.npy"))
# where did I get the brain masks initially?

#mask_arr = (np.mean(M03_minus_M00, axis=0) >= 0.1) & (np.std(M03_minus_M00, axis=0) >= 1e-6)
# mask_arr = mask_arr & GM_brain_mask
mask_arr = mask_arr.astype(bool)
print("Mask shape", mask_arr.shape)

# get and save the mask as a nifti file
mask_img = nibabel.Nifti1Image(mask_arr.astype(float), affine=ref_img.affine)
mask_img.to_filename(os.path.join(OUTPUT_DIR, "mask.nii.gz"))



def mask_and_save(Y, mask, filename):
    """Mask images in Y with mask, then save the images in the OUTPUT_DIR folder as a nifti file.
    
    Arguments:
        - Y: 4D numpy array containing 3D images (either the difference or the original ones)
        - mask: mask to reduce the mean and std to the grey matter regions.
        - filename: name of the saved nifti file containing the cs images."""
    
    Ycs = Y[:, mask]
    print("Ycs shape", Ycs.shape)
    assert np.count_nonzero(np.isnan(Ycs)) == 0  # check that no more nan in the data
    assert Ycs.shape == (Y.shape[0], np.sum(mask))

    # convert the images as niftii images
    images_cs = []
    for i in range(Ycs.shape[0]):
        arr = np.zeros(Y[0].shape)
        arr[mask] = Ycs[i, :]
        images_cs.append(nibabel.Nifti1Image(arr, affine=ref_img.affine))
    
    print("Save images")
    ALLcs = nilearn.image.concat_imgs(images_cs)
    ALLcs.to_filename(os.path.join(OUTPUT_DIR, f"{filename}.nii.gz"))
    
    return Ycs


Ybrain = mask_and_save(M03_minus_M00, mask_arr, filename='M03_minus_M00')

########################################################################################################################
# Create design matrices
print("Create design matrix to take into account age, sex and/or sites")

# reshape one-hot encoding of sites
sites_matrix = np.concatenate(pop["site.num"].values).reshape((n_sub, n_sites-1))


# create the design matrix

# list of usable Design matrices
# M03-M00 ~ 1
onecol_Design = np.ones((n_sub,1))

# M03-M00 ~ 1+age
age_Design = np.ones((n_sub, 2))
age_Design[:, 1] = pop.age

# M03-M00 ~ 1+sex
sex_Design = np.ones((n_sub, 2))
age_Design[:, 1] = pop["sex.num"]

# M03-M00 ~ 1+age+sex
no_site_Design = np.zeros((n_sub, 1+2))
no_site_Design[:, 0] = 1  # intercept
# age and sex
no_site_Design[:, 1:3] = pop[["age", "sex.num"]]

# M03-M00 ~ 1+age+sex+site
full_Design = np.zeros((n_sub, 1+2+n_sites-1))
full_Design[:, 0] = 1  # intercept
# age and sex
full_Design[:, 1:3] = pop[["age", "sex.num"]]
# sites
full_Design[:, 3:] = sites_matrix

# select the used design matrix
Design = full_Design

print("Design matrix shape and head:\n", Design.shape)
print(Design[:5])
print("Save design matrix")
np.savetxt(os.path.join(OUTPUT_DIR, 'design_matrix.txt'), Design, fmt='%d')

########################################################################################################################
# Statistics Voxel Based
print("VOXEL BASED STATISTICS USING MULM")


def univar_stats(Y, X, contrasts, path_prefix, mask_img, threspval=10**-3, threstval=3,
                 vmax=15, two_tailed=False):
    """Compute the t-test voxel wise for images in Y with X the design matrix.
    """

    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

    # save contrasts used
    with open(path_prefix+"_contrasts.txt", 'w') as file:
        for contrast in contrasts:
            file.write(str(contrast)+'\n')

    mod = mulm.MUOLS(Y, X)
    tvals, pvals, df = mod.fit().t_test(contrasts, pval=True, two_tailed=two_tailed)

    for i in range(len(contrasts)):
        print([[thres, np.sum(pvals <thres), np.sum(pvals <thres)/pvals.size] for thres in 10. ** np.array([-4, -3, -2])])

        tstat_arr = np.zeros(mask_arr.shape)
        pvals_arr = np.zeros(mask_arr.shape)

        pvals_arr[mask_arr] = -np.log10(pvals[i])
        tstat_arr[mask_arr] = tvals[i]

        pvals_img = nibabel.Nifti1Image(pvals_arr, affine=mask_img.affine)
        pvals_img.to_filename(path_prefix + f"_{str(i)}_vox_p_tstat-mulm_log10.nii.gz")

        tstat_img = nibabel.Nifti1Image(tstat_arr, affine=mask_img.affine)
        tstat_img.to_filename(path_prefix + f"_{str(i)}_tstat-mulm.nii.gz")

        # compute the associated p value of the vmax used for t
        vmax_p_univar = -np.log10(scipy.stats.t.sf(vmax, df=Design.shape[0]-3))

        fig = plt.figure(figsize=(26,  10))
        ax = fig.add_subplot(211)
        ax.set_title("-log pvalues >%.2f"%  -np.log10(threspval))
        plotting.plot_glass_brain(pvals_img, threshold=-np.log10(threspval), figure=fig, axes=ax,
                                  colorbar=True, vmax=vmax_p_univar)

        ax = fig.add_subplot(212)
        ax.set_title("T-stats with T>%.2f" % threstval)
        plotting.plot_glass_brain(tstat_img, threshold=threstval, figure=fig, axes=ax, colorbar=True,
                                  plot_abs=False, vmin=-vmax, vmax=vmax, cmap=twoways_cmaps[0])

        plt.savefig(path_prefix +  f"_{str(i)}_tstat-mulm.png")

    return tstat_arr, pvals_arr


# set the p and t threshold
threspval = 5e-2
threstval = np.abs(scipy.stats.t.ppf(threspval / 2, df=Design.shape[0]-3))
vmax_t_univar = 5*threstval

# define the contrasts used
univar_contrasts = [[1] + [0]*(Design.shape[1] - 1)]

# define the folder where to save the univar analysis
path_prefix=os.path.join(OUTPUT_DIR, "univar_stats", MODEL_NAME)

# 2 sided test
tmap, pmap = univar_stats(Y=Ybrain, X=Design, contrasts=univar_contrasts,
                          path_prefix=path_prefix,
                          mask_img=mask_img, threspval=threspval, threstval=threstval,
                          vmax=vmax_t_univar, two_tailed=True)


########################################################################################################################
# Univar stats glassviews
print("Generate glassviews for mulm univar stats")


def plot_glassview(image_path, save_path, mask=None, cmap='autumn', vmin=None,
                   vmax=None, verbose=False):
    """Plot the glassview of an image saved as nifti using nilearn's plot_glass_brain.
    
    Arguments:
        - image_path: path to the nifti image to be plotted.
        - save_path: path where the glassview will be saved.
        - mask: numpy array used to mask the image OR float used as threshold.
        If None, then an automatic thresholding is applied by plot_glass_brain.
        - cmap: matplotlib color map
        - vmin: minimal value of the color map. /!\ Must be negative if the image 
        is two sided
        - vmax: maximal value of the color map."""
    
    if verbose: print("Plot glassview")
    map_img = nibabel.load(image_path)
    
    if mask is None:
        # not filtered glassview
        threshold = 'auto'
        saved_mapped_img = map_img
    elif type(mask) == float:
        if verbose: print(f"Apply thresholding at 1-p={mask}")
        threshold = mask
        saved_mapped_img = map_img
    else:
        # mask is a 3D numpy array to mask with
        if verbose: print("Apply masking")
        map_arr = map_img.get_fdata()
        if verbose: print("Number of non zero voxels before masking", np.count_nonzero(map_arr))
        map_arr[~mask] = 0
        if verbose: print("Number of non zero voxels after masking", np.count_nonzero(map_arr))
        saved_mapped_img = nibabel.Nifti1Image(map_arr, affine=ref_img.affine)
        threshold = 'auto'


    # plot the figure
    fig = plt.figure(figsize=(13.33, 4 * 1))
    plotting.plot_glass_brain(saved_mapped_img, colorbar=True, cmap=cmap, vmin=vmin, vmax=vmax,
                              figure=fig, plot_abs=False, threshold=threshold)
                              #, figure=fig, axes=ax)
    plt.savefig(save_path)
    plt.close(fig)



#############################################################################
# FSL vbm(tfce)
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise

print("TFCE ANALYSIS WITH FSL")

# def run_TFCE_analysis(Design, contrasts, input_data_path, fsl_path):
#     """Create the right contrast (and group) matrices. Save all the matrices in .mat
#     format. Then run the randomize command to run TFCE. The MRI data is got from an
#     already saved file."""

#     os.makedirs(fsl_path, exist_ok=True)
#     prefix = os.path.join(fsl_path, MODEL_NAME)

#     # save the Design matrix as a .mat file
#     # pd.DataFrame(Design).to_csv(prefix +'_design.txt', header=None, index=None, sep=' ', mode='a')
#     # Text2Vest.main([prefix +'_design.txt', prefix +'_design.mat'])
#     # os.remove(prefix +'_design.txt')

#     # save the contrast matrix as a .mat file 
#     # np.savetxt(prefix +'_contrast.txt', contrasts, fmt='%i')
#     # Text2Vest.main([prefix +'_contrast.txt', prefix +'_contrast.mat'])
#     # os.remove(prefix +'_contrast.txt')

#     # create the command line to be run
#     # cmd = ["randomise", '-i', input_data_path, "-m",  os.path.join(OUTPUT_DIR, "mask.nii.gz"),
#     # "-o", prefix,
#     # '-d', prefix +'_design.mat',
#     # '-t', prefix +'_contrast.mat',
#     # '-T', '-C', '1',
#     # '-n', '500',
#     # '-R', '--uncorrp', '-N']

#     # run the command line with FSL
#     # print("Command line:\n".join(cmd))
#     # runfsl(cmd)


# list of usable contrasts
age_contrast = [0,1] + [0]*(Design.shape[1] - 2)
minus_age_contrast = [0,-1] + [0]*(Design.shape[1] - 2)
sex_contrast = [0,0,1] + [0]*(Design.shape[1] - 3)
sites_contrast = [0,0,0] + [1]*(Design.shape[1] - 3)
full_contrast = [1]*Design.shape[1]

# contrasts actually used for FSL
#tfce_contrasts = np.array([1])

input_data_path = os.path.join(OUTPUT_DIR, "M03_minus_M00.nii.gz")



nifti_masker = NiftiMasker(mask_img=mask_img ,smoothing_fwhm=5, memory="nilearn_cache", memory_level=1)
nifti_masker.fit(input_data_path)


X = Design
Y = Ybrain

print("xshape",X.shape)
contrasts = np.identity(X.shape[1])

mod = mulm.MUOLS(Y, X)
tvals, rawp, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

neg_log10_tfce_pvals = mod.t_test_tfce(contrasts, mask=mask_img,nperms=10,two_tailed=True )

neg_log_pvals_tfce_unmasked = nifti_masker.inverse_transform(
    neg_log10_tfce_pvals[0, :]
)

print("on est allé au bout")


# Plot the final results.
z_slice = 12  # Plotted slice
threshold = -np.log10(0.1)  # 10% corrected
vmax = np.amax(neg_log10_tfce_pvals)

images_to_plot = {
    "Permutation Test (auditory)\n(Max TFCE FWE)": neg_log_pvals_tfce_unmasked,
}

fig, ax = plt.subplots(figsize=(14, 5), ncols=1)

for i_col, (title, img) in enumerate(images_to_plot.items()):
    n_detections = (get_data(img) > threshold).sum()
    new_title = f"{title}\n{n_detections} sig. voxels"

    plotting.plot_glass_brain(
        img,
        colorbar=True,
        vmax=vmax,
        plot_abs=False,
        cut_coords=[12],
        threshold=threshold,
        figure=fig,
        axes=ax,
    )
    ax.set_title(new_title)

fig.suptitle(
    "Group left button press ($-\\log_{10}$ p-values)",
    y=1.3,
    fontsize=16,
)

plt.show()

