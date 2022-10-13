import numpy as np

# location of processed info
# center info will be saved to figure_dir
center_filename = "centers_data.csv"
crop_filename = "crop_data.csv"
# figure_dir = './figures/test_max_projection_only'
# crop_dir = './figures/crops_max_projection_info_with_masks'
figure_dir = "./figures/test_crop_new"
crop_dir = "./figures/crops_info_with_masks"
mask_folder_name = "masks"  # name of mask folder in crop_dir

# if we only want to generate crop but not to find all signal candidates
gen_crop_only = False
# if only use max projection for finding stage 1 centers
process_max_projection_only = True

bounding_size = 10  # for averaging each pixel values. can be disregarded
collapse_dist_threshold = 10  # minimum distance between centers
collapse_dist_threshold_3d = 15
# whether generate plot for each plot for debug. False during production phase
save_fig = True
max_center_num_per_2d_image = 7  # how many centers per crop?
# max distance of two points with same track id in consecutive time frame
tracking_dist_threshold = 20
tracking_gap = 15  # max gaps allowed when map signals to previous time point signals
# how many points to consider as center candidates according to Gaussian
# distribution?
sampling_quantile_per_image = 0.9
# for processing simple background information of a center
background_boundings = [30, 30, 2]
# filtering centers: how many standard deviations should each center pixel
# value be larger than background average? (excluding signal area)
filter_contrast_factors = np.linspace(0, 5, 5)
min_traj_len = 7  # minimum trajectory to be considered as a true signal
traj_circle_radius = 5  # radius of point in trajectory

cell_radius = 200  # cell radius used in filtering signals
max_allowed_signals_per_cell = 3
