import numpy as np

def compute_intrinsics(img_size, fov):
    W, H = img_size
    FoV_x = fov / 180 * np.pi  # to rad
    f = 1 / np.tan(FoV_x / 2) * (W / 2)

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])
    return K

def normalize(v):
    return v / np.linalg.norm(v)

def camera_pose(eye, front, up):
    # print('eye', eye)
    # print('front', front)
    # print('up', up)
    z = normalize(-1 * front) # -1 except for mesh
    x = normalize(np.cross(up, z))
    y = normalize(np.cross(z, x))
 
    # convert to col vector
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    eye = eye.reshape(-1, 1)

    pose = np.block([
        [x, y, z, eye],
        [0, 0, 0, 1]
    ])
    return pose

def compute_extrinsics(elevation_rad, azimuth_rad, radius):
    # elevation_rad += 1e-10
    azimuth_rad = np.pi - azimuth_rad
    e = np.array([
        radius * np.cos(elevation_rad) * np.cos(azimuth_rad),
        radius * np.cos(elevation_rad) * np.sin(azimuth_rad),
        radius * np.sin(elevation_rad)
    ])
    up = np.array([0,0,1])
    pose = camera_pose(e, -e, up)
    world_2_cam = np.linalg.inv(pose)
    return world_2_cam[:3,:]