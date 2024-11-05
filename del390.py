import os
import sys

path = sys.argv[1]
threshold = int(sys.argv[2])
dir_dic = {}

# Loop through each directory in the provided path
for class_dir in os.listdir(path):
    # Full path to the subdirectory
    dir_path = os.path.join(path, class_dir)
    if class_dir == ".DS_Store":
        continue
    class_dir = int(class_dir)
    num_dir = {}

    # Only proceed if it's a directory
    if os.path.isdir(dir_path):
        i = 0
        listdir = os.listdir(dir_path)
        listdir.sort()
        for file in listdir:
            if i >= threshold:
                os.remove(os.path.join(dir_path, file))
            i += 1

# sort dir
sort_dir_dic = sorted(dir_dic.keys())

# Print the dictionary
for key in sort_dir_dic:
  print(f"{dir_dic[key]["n"]}")
