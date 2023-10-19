#!/bin/bash

# Set the path to the folder containing the zip files
zip_folder="./raw-zips"

# Loop through all zip files in the folder and extract them
for zip_file in "$zip_folder"/*.zip; do
    unzip -q "$zip_file" -d "${zip_file%.*}"
    mv "${zip_file%.*}" "$(basename "${zip_file%.*}")"
done