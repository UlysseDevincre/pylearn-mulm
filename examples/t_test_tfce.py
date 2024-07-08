
"""
Mass-univariate analysis with TFCE
==================================
This example demonstrates a mass-univariate analysis using TFCE
through the MULM library.

The dataset comprises 94 subjects from the Localizer dataset.
The task identifies brain regions activated when subjects
press a button with their left hand following an auditory cue.
"""
# %%
################################################################################
# Import necessary libraries
# --------------------------
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script requires the matplotlib library")

import numpy as np
from nilearn import datasets, plotting
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask
from nilearn.image import get_data, threshold_img
from nilearn.mass_univariate._utils import normalize_matrix_on_axis
import nibabel as nib
import mulm
from mulm.utils import ttest_tfce

################################################################################
# Download localizer dataset
# --------------------------
# Fetching the data for 94 subjects from the Localizer dataset.
n_samples = 94 
localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left button press (auditory cue)"],
    n_subjects=n_samples,
    legacy_format=False,
)

# Load the behavioral data and remove NaN values.
tested_vars = localizer_dataset.ext_vars["pseudo"]
mask_quality_check = np.where(np.logical_not(np.isnan(tested_vars)))[0]
n_samples = mask_quality_check.size

# Get the contrast maps corresponding to the dataset.
contrast_map_filenames = [
    localizer_dataset.cmaps[i] for i in mask_quality_check
]
tested_vars = tested_vars[mask_quality_check].values.reshape((-1, 1))






################################################################################
# Plot the contrast map
# ---------------------
# Display the first contrast map and apply a threshold.
# The contrast map is a 3D image used to perform statistical tests
# on neuroimaging data, showing brain region activation.

img = nib.load(contrast_map_filenames[0])
plotting.plot_stat_map(img)
plotting.show()

img_thresh = threshold_img(img, 3.0, two_sided=True)
plotting.plot_stat_map(img_thresh)
plotting.show()





################################################################################
# Preprocess the neuroimaging data
# --------------------------------
# Getting the mask to the contrast maps then transform
# it into 2D numpy arrays with the mask applied.

nifti_masker = NiftiMasker(smoothing_fwhm=5, memory="nilearn_cache", memory_level=1)
mask = nifti_masker.fit(contrast_map_filenames).mask_img_


# Transform the contrast maps in 2D numpy arrays (les donnée sont modifiées)

target_vars = []
for i in range(n_samples):
    data = nib.load(contrast_map_filenames[i]).get_fdata()
    masked_data = data[mask.get_fdata() == 1]
    target_vars.append(masked_data)
target_vars = np.array(target_vars)

#Transform the contrast maps in 2D numpy arrays (les donnée ne sont pas modifiées av ec les masker nifti)

target_vars2 = nifti_masker.fit_transform(contrast_map_filenames, mask)








# Normalize the variables on the axis.

targetvars_resid_covars = normalize_matrix_on_axis(target_vars).T
testedvars_resid_covars = normalize_matrix_on_axis(tested_vars).copy()





################################################################################
# Perform the statistical test: Student's t-test
# ----------------------------------------------
# Conduct t-tests on all regressors and apply Bonferroni correction
# to correct for multiple comparisons.

X = testedvars_resid_covars
Y = targetvars_resid_covars.T

print("xshape",X.shape)
contrasts = np.identity(X.shape[1])

mod = mulm.MUOLS(Y, X)
tvals, rawp, df = mod.fit().t_test(contrasts, pval=True, two_tailed=True)

print(tvals.shape)
print(tvals)

# Apply Bonferroni correction for multiple comparisons.
num_comparisons = len(rawp)
alpha = 0.05
bonferroni_correction = alpha / num_comparisons
pvals_corrected = rawp * num_comparisons

# Create a statistical map of corrected p-values.
p_img = np.zeros(mask.shape)
for i in range(contrasts.shape[0]):
    p_img[mask.get_fdata() == 1] = pvals_corrected[i]
p_img = nib.Nifti1Image(p_img, mask.affine)
plotting.plot_stat_map(p_img)
plotting.show()

# Convert the t-values into t-maps (niimg).
scores_4d = np.zeros(mask.shape + (contrasts.shape[0],))
for i in range(contrasts.shape[0]):
    scores_4d[mask.get_fdata() == 1, i] = tvals[i]
scores_4d_m = nib.Nifti1Image(scores_4d, mask.affine)
plotting.plot_stat_map(scores_4d_m)
plotting.show()



################################################################################
# Perform the TFCE
# ----------------
# Execute Threshold-Free Cluster Enhancement (TFCE) on the data.

tfce_original_data = ttest_tfce(
    scores_4d,
    bin_struct=None,
    two_sided_test=True
)





################################################################################
# Plot the TFCE results
# ---------------------
# Return the t-map to a niimg and plot the TFCE results with a chosen threshold.

tfce_original_data = nib.Nifti1Image(tfce_original_data, mask.affine, mask.header)


image_data = tfce_original_data.get_fdata()
mask_data = mask.get_fdata()
mask_data_bool = mask_data.astype(bool)

# Apply the mask by element-wise multiplication


tfce_original_data = nib.Nifti1Image(image_data, mask.affine, mask.header)

plotting.plot_stat_map(tfce_original_data)
plotting.show()

# Empirically chosen threshold.
threshold = 100000
thresholded_tfce_original_data = threshold_img(tfce_original_data, threshold)
plotting.plot_stat_map(thresholded_tfce_original_data)
plotting.show()


################################################################################
# Perform the statistical test: Student's t-test with TFCE
# --------------------------------------------------------
# Conduct t-tests on all regressors and apply TFCE to correct for multiple comparisons.

neg_log10_tfce_pvals = mod.t_test_tfce(contrasts, mask=mask,nperms=100,two_tailed=True )
neg_log_pvals_tfce_unmasked = nifti_masker.inverse_transform(
    neg_log10_tfce_pvals[0, :]
)

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
        display_mode="z",
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

# %%