import os
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    """Returns a list of .xz files in the given directory."""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

# Specify the folder path and output filenames

folder_path = r"C:\Users\Saikiran\Downloads\data"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"
split_files = int(input("How many files would you like to split this into? "))

# Retrieve .xz files from the directory
files = []
for filename in os.listdir(folder_path):
    files.append(filename)
total_files = len(files)
max_count = total_files // split_files if split_files != 0 else total_files

vocab = set()

# Process training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files[:max_count], total=min(max_count, total_files)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()  # Read the contents of the file
            outfile.write(text)  # Write to the output file
            characters = set(text)  # Get unique characters from the text
            vocab.update(characters)  # Update vocabulary

# Process validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files[max_count:], total=total_files - max_count):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()  # Read the contents of the file
            outfile.write(text)  # Write to the output file
            characters = set(text)  # Get unique characters from the text
            vocab.update(characters)  # Update vocabulary

# Write vocabulary to file
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')  # Write each character on a new line
