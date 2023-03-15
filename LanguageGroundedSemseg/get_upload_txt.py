import os
import sys

if __name__ == '__main__':
    data_dir = sys.argv[1]
    for root, dirs, files in os.walk(data_dir):
        for file_name in files:
            if "txt" not in file_name:
                os.remove(os.path.join(data_dir, file_name))