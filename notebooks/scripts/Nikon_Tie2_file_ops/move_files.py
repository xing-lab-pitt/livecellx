import os
import shutil

# Set the source directory path
src_dir = "./tifs_CFP_A549_VIM_120hr_NoTreat_NA_YL_Ti2e_2023-03-22"

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    print("filename:", filename)
    # Check if the file is a regular file
    if os.path.isfile(os.path.join(src_dir, filename)):
        # Extract the XY number from the filename
        xy = filename.split("_XY")[1][:2]
        # Create the XY subdirectory if it doesn't exist
        xy_dir = os.path.join(os.getcwd(), f"XY{xy}")
        if not os.path.exists(xy_dir):
            os.makedirs(xy_dir)
        # Determine whether the file is DIC or TRITC
        if "DIC" in filename:
            subdir = "DIC"
        elif "TRITC" in filename:
            subdir = "TRITC"
        else:
            subdir = None
        # Move the file to the appropriate subdirectory
        if subdir is not None:
            subdir_path = os.path.join(xy_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
            shutil.move(os.path.join(src_dir, filename), os.path.join(subdir_path, filename))
        else:
            shutil.move(os.path.join(src_dir, filename), xy_dir)
