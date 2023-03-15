import os
import sys

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    split_num = int(sys.argv[2])

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    if not os.path.exists(val_dir): os.makedirs(val_dir)
    train_txt = os.path.join(dataset_dir, "train.txt")
    val_txt = os.path.join(dataset_dir, "val.txt")

    file_names = sorted(os.listdir(train_dir))
    ply_files = []
    for file_name in file_names:
        if "ply" in file_name:
            ply_files.append(file_name)

    val_ply_files = ply_files[len(ply_files) - split_num:]
    train_ply_files = ply_files[:len(ply_files) - split_num]

    # move file to val/
    for i, val_ply_file in enumerate(val_ply_files):
        os.popen(f'mv {os.path.join(train_dir, val_ply_file)} {val_dir}')

    # modify train.txt & val.txt
    with open(train_txt, "w") as f:
        for i, train_ply_file in enumerate(train_ply_files):
            f.write(f"train/{train_ply_file}\n")
    with open(val_txt, "w") as f:
        for i, val_ply_file in enumerate(val_ply_files):
            f.write(f"val/{val_ply_file}\n")
