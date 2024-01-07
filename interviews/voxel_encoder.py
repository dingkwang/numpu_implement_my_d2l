import itertools
from collections import defaultdict
from email.policy import default

import torch

# def voxelization(
#     point_cloud,
#     voxel_grid_size=[0.5, 0.5, 0.5],
#     pc_range=[0, -72, -3.0, 144, 72, 5.0],  # (-x, -y, -z, x, y, z)
# ):
#     # Normalize the point cloud if necessary
#     # point_cloud = normalize(point_cloud)

#     # Compute voxel indices
#     voxels = defaultdict(list)
#     voxel_indices = np.floor(point_cloud[:, :3] / voxel_grid_size).astype(int)

#     x_grid = [
#         x_i for x_i in range(int((pc_range[3] - pc_range[0]) / voxel_grid_size[0]) + 1)
#     ]
#     y_grid = [
#         y_i for y_i in range(int((pc_range[4] - pc_range[1]) / voxel_grid_size[1]) + 1)
#     ]
#     z_grid = [
#         x_i for x_i in range(int((pc_range[5] - pc_range[2]) / voxel_grid_size[2]) + 1)
#     ]

#     for combination in itertools.product(x_grid, y_grid, z_grid):
#         voxels[combination]
#     num_points_per_voxel_out = [0] * len(voxels)

#     # Create a dictionary to store voxel data
#     voxels = {}
#     for i, voxel_index in enumerate(voxel_indices):
#         key = tuple(voxel_index)
#         if key not in voxels:
#             import ipdb

#             ipdb.set_trace()

#             voxels[key] = []
#         voxels[key].append(point_cloud[i])
#         i = voxel_indices[3] +
#         num_points_per_voxel_out[voxel_indices]

#     # Compute features for each voxel
#     # voxel_features = {}
#     # for key, points in voxels.items():
#     #     points = np.array(points)
#     #     feature = {
#     #         "center_of_mass": np.mean(points[:, :3], axis=0),
#     #         "average_reflectivity": np.mean(points[:, 3]),
#     #     }
#     #     voxel_features[key] = feature
#     voxels_out = list(voxels.values())
#     coors_out = list(voxels.keys())

#     return voxels_out, coors_out, num_points_per_voxel_out


def voxelization(point_cloud, voxel_size, max_num_points_per_voxel):
    # Compute voxel indices
    voxel_indices = torch.floor(point_cloud[:, :3] / voxel_size).int()

    # Dictionary to store voxel data
    voxels = {}
    for i, voxel_index in enumerate(voxel_indices):
        key = tuple(voxel_index)
        if key not in voxels:
            voxels[key] = []
        if len(voxels[key]) < max_num_points_per_voxel:
            voxels[key].append(point_cloud[i])

        # Limit the number of points per voxel

    # Prepare output arrays
    N = len(voxels)
    features_dim = point_cloud.shape[1]
    voxels_out = torch.zeros((N, max_num_points_per_voxel, features_dim))
    coors_out = torch.zeros((N, 3), dtype=torch.int)
    num_points_per_voxel_out = torch.zeros(N, dtype=int)

    for i, (voxel_index, points) in enumerate(voxels.items()):
        coors_out[i] = torch.tensor(voxel_index)
        num_points = len(points)
        num_points_per_voxel_out[i] = num_points

        voxels_out[i, :num_points] = torch.stack(points)

    return voxels_out, coors_out, num_points_per_voxel_out


def hard_simple_vfe(
    num_points,
    features,
    num_features=4,
):
    # Step 1: Selecting Features
    selected_features = features[
        :, :, :num_features
    ]  # Shape: (N, M, self.num_features)

    # Step 2: Summing Features

    summed_features = selected_features.sum(axis=1)  # Shape: (N, self.num_features)

    # Step 3: Converting num_points Type
    num_points_converted = num_points.type_as(features)  # Shape: (N,)

    # Step 4: Reshaping num_points
    num_points_reshaped = num_points_converted.view(-1, 1)  # Shape: (N, 1)

    # Step 5: Dividing Summed Features by num_points
    points_mean = summed_features / num_points_reshaped  # Shape: (N, self.num_features)

    return points_mean.contiguous()


N = 1000
point_cloud = torch.rand(1000, 4)
point_cloud[:, :3] = point_cloud[:, :3] * 1000

voxel_grid_size = 0.5  # Define the voxel size
voxels, coors, num_points = voxelization(
    point_cloud, voxel_grid_size, max_num_points_per_voxel=10
)

print("voxels.shape", voxels.shape)
print("coors.shape", coors.shape)
print("num_points.shape", num_points.shape)

voxel_features = hard_simple_vfe(
    num_points=num_points,
    features=voxels,
    num_features=4,
)
print("voxel_features.shape", voxel_features.shape)

import ipdb

ipdb.set_trace()
