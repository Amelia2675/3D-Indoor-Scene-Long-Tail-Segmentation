import sys
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    plt_filepath = sys.argv[1]
    pcd = o3d.io.read_point_cloud(plt_filepath)
    o3d.visualization.draw_geometries([pcd])
