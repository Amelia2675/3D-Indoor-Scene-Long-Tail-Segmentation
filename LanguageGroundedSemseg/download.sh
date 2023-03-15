#!/bin/bash

# download checkpoint
gdown -O "checkpoint_best.ckpt" "https://drive.google.com/uc?id=1WC-FvzUTkc_zmcmzFPMw4GrwD71pxNRy&confirm=t"

# download dataset
gdown -O "dataset.zip" "https://drive.google.com/uc?id=1kBwyiqMEEmbFfUsQy6FyNAInD_xZW0Ym&confirm=t"
unzip dataset.zip