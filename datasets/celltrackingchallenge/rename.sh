#!/bin/bash

# Set the path to the folder containing the files
folder="./raw-zips"

# Rename all files in the folder that contain " (1)" and replace it with "-test"
rename 's/ \(1\)/-test/' "$folder"/*" (1)"*