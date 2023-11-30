#!/bin/bash

# Define the original and new substrings
target_dir="."
original_substr=""
new_substr=""

# Replace the substrings in all files in the folder recursively
count=$(find $target_dir -type f -name "*.json" -exec sed -i "s|$original_substr|$new_substr|g" {} + | wc -l)

echo "Replaced $count occurrences of '$original_substr' with '$new_substr'"