import os
import json
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description="Replace all occurrences of a substring in JSON files in a subdirectory.")
parser.add_argument("dir_path", type=str, help="the path to the subdirectory")
parser.add_argument("old_substring", type=str, help="the substring to replace")
parser.add_argument("new_substring", type=str, help="the new substring")

# Parse the command line arguments
args = parser.parse_args()

# Loop over all files in the subdirectory
for root, dirs, files in os.walk(args.dir_path):
    for file in files:
        # Check if the file is a JSON file
        if file.endswith(".json"):
            print("Processing file: " + file)
            # Read the contents of the file
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                contents = json.load(f)

            # Replace all occurrences of the old substring with the new substring
            new_contents = json.dumps(contents).replace(args.old_substring, args.new_substring)

            # Write the updated contents back to the file
            with open(file_path, "w") as f:
                f.write(new_contents)
