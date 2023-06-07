#!/bin/bash

source_directory="$1"
destination_directory="$2"
interval="$3"

# Check if all arguments are provided
if [[ -z "$source_directory" || -z "$destination_directory" || -z "$interval" ]]; then
    echo "Usage: bash script.sh <source_directory> <destination_directory> <interval>"
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$destination_directory"

# Change to the source directory
cd "$source_directory"

# Sort the TIF files by name
sorted_files=($(ls -1 *.tif | sort))
total_files="${#sorted_files[@]}"

# Calculate the number of files to move
num_files=$((total_files / interval))

# Loop through the sorted files
for ((i=0; i<num_files; i++)); do
    index=$((i * interval))
    file="${sorted_files[index]}"
    mv "$file" "$destination_directory"
    echo "Moved file: $file"
done