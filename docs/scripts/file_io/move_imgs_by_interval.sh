#!/bin/bash

source_directory="$1"
destination_directory="$2"
total_images="$3"

# Check if all arguments are provided
if [[ -z "$source_directory" || -z "$destination_directory" || -z "$total_images" ]]; then
    echo "Usage: bash script.sh <source_directory> <destination_directory> <total_images>"
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$destination_directory"

# Change to the source directory
cd "$source_directory"

# Sort the TIF files by name
sorted_files=($(ls -1 *.tif | sort))
total_files="${#sorted_files[@]}"

# Calculate the interval based on the total number of files
interval=$((total_files / total_images))
count=0

# Loop through the sorted files
for file in "${sorted_files[@]}"; do
    ((count++))
    if ((count % interval == 0)); then
        mv "$file" "$destination_directory"
        echo "Moved file: $file"
    fi
done