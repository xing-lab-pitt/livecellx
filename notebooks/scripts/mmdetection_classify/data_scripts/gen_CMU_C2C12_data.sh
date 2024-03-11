#!/bin/bash

# Set the environment variables
export EXP_DIR="090303-C2C12P15-FGF2"
export ANNOTATION_DIR="/home/ken67/LiveCellTracker-dev/datasets/CMU_C2C12/$EXP_DIR/Annotation_Human"

# Create output directory if it doesn't exist
mkdir -p "../datasets/mitosis-annotations-2023/CMU_C2C12_v16/$EXP_DIR/"

# Loop through the files F0001 to F0018
for i in {1..18}; do
    # Adjust the file number format based on the value of i
    if [ $i -lt 10 ]; then
        FILE_NUM="F000${i}" # For numbers 1-9, use three leading zeros
    else
        FILE_NUM="F00${i}"  # For numbers 10-18, use two leading zeros
    fi

    XML_PATH="${ANNOTATION_DIR}/Human exp1_${FILE_NUM} Data.xml"
    IMG_DIR="/home/ken67/LiveCellTracker-dev/datasets/CMU_C2C12/$EXP_DIR/exp1_${FILE_NUM}"

    # Run the python script for the current file
    python ../livecellx/track/process_annotation_CMU_C2C12.py \
        --xml_path="$XML_PATH" \
        --out_dir="/home/ken67/LiveCellTracker-dev/datasets/mitosis-annotations-2023/CMU_C2C12_v16/$EXP_DIR/" \
        --img_dir="$IMG_DIR"
done
