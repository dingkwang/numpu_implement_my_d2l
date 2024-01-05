import numpy as np


def voxelization(point_cloud, voxel_size):
    # Normalize the point cloud if necessary
    # point_cloud = normalize(point_cloud)

    # Compute voxel indices
    voxel_indices = np.floor(point_cloud[:, :3] / voxel_size).astype(int)

    # Create a dictionary to store voxel data
    voxels = {}
    for i, voxel_index in enumerate(voxel_indices):
        key = tuple(voxel_index)
        if key not in voxels:
            voxels[key] = []
        voxels[key].append(point_cloud[i])

    # Compute features for each voxel
    voxel_features = {}
    for key, points in voxels.items():
        points = np.array(points)
        feature = {
            "center_of_mass": np.mean(points[:, :3], axis=0),
            "average_reflectivity": np.mean(points[:, 3]),
        }
        voxel_features[key] = feature
        import ipdb

        ipdb.set_trace()

    return voxel_features


# Example usage
# point_cloud = np.array(
#     [[0.5, 1.2, 3.3, 0.8], [0.7, 1.1, 3.1, 0.6], ...]
# )  # Your point cloud data

N = 1000
point_cloud = np.random.rand(N, 4)
point_cloud[:, :3] = point_cloud[:, :3] * 10

voxel_size = 0.5  # Define the voxel size
voxelized_data = voxelization(point_cloud, voxel_size)
