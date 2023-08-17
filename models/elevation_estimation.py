import numpy as np
from PIL import Image
import cv2
import itertools
import time
from scipy import spatial
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import torch
import os, sys
sys.path.append("..")
from scripts.utils import *
from scripts.zero123_utils import *

class ElevationEstimation():
    def __init__(self, 
                 spherical_coordinates, 
                 image_paths, 
                 fov=50.0,
                 radius=2.5,
                 image_dir=".", 
                 max_size=320,
                 actual_spherical_coordinates=(0,0)
                 ):
        self.matcher = KF.LoFTR(pretrained="indoor_new")

        self.images = []
        for i,file in enumerate(image_paths):
            img = Image.open(file)
            img = convert_to_rgb(img)
            img = Image.fromarray(crop_to_square(np.array(img), size=max_size))
            self.img_size = img.size
            filename = os.path.join(image_dir, f"{i}.jpg")
            img.save(filename)
            self.images.append(filename)

        self.fov = fov
        self.radius = radius
        self.spherical_coordinates = [(*k, self.radius) for k in spherical_coordinates]
        self.actual_elevation, self.actual_azimuth = actual_spherical_coordinates

    # Main function
    def estimate_elevation(self, elevation_range=range(-30, 31, 10), plot=False, verbose=False):

        tic = time.time()

        # Iterate through all image pairs (for feature matching) and image triplets (for reprojection)
        print(f"Feature matching...", end="")
        matched_features = {}
        for pair in itertools.combinations(range(len(self.images)), 2):
            i1, i2 = pair
            matched_features[pair] = self.feature_matching(self.images[i1], self.images[i2])
        print("Done.")

        # Iterate through rough elevation candidates
        errors_for_each_elevation = []
        for elevation in elevation_range:
            error = self.calculate_total_error_for_elevation(matched_features, elevation, plot=plot)
            errors_for_each_elevation.append(error)

        sorted_indices = np.argsort(errors_for_each_elevation)
        best_rough_elevation = elevation_range[sorted_indices[0]]
        print(f"Best rough elevation = {best_rough_elevation}")
        if verbose:
            print()
            for ind in sorted_indices[:15]:
                print(f"{elevation_range[ind]}: {errors_for_each_elevation[ind]}")
            print()

        # Iterate through fine elevation candidates
        errors_for_each_elevation = []
        fine_elevation_range = [best_rough_elevation-5, best_rough_elevation, best_rough_elevation+5]
        for elevation in fine_elevation_range:
            error = self.calculate_total_error_for_elevation(matched_features, elevation, plot=plot)
            errors_for_each_elevation.append(error)

        sorted_indices = np.argsort(errors_for_each_elevation)
        best_elevation = fine_elevation_range[sorted_indices[0]]
        print(f"Best elevation = {best_elevation}")

        toc = time.time()
        print(f"Elapsed time: {(toc-tic):.2f}s")

        # delete saved image files
        for img in self.images:
            os.remove(img)

        return best_elevation

    def feature_matching(self, img_name1, img_name2, plot=False):
        img1 = K.io.load_image(img_name1, K.io.ImageLoadType.RGB32)[None, ...]
        img2 = K.io.load_image(img_name2, K.io.ImageLoadType.RGB32)[None, ...]

        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img2),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()

        # sometimes cv2 throws assertion error
        try:
            Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.99, 100000)
            inliers = inliers > 0
        except:
            print(len(mkpts0), len(mkpts1))
            return mkpts0.astype(int), mkpts1.astype(int)

        if plot:
            draw_LAF_matches(
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts0).view(1, -1, 2),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts0.shape[0]).view(1, -1, 1),
                ),
                KF.laf_from_center_scale_ori(
                    torch.from_numpy(mkpts1).view(1, -1, 2),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                    torch.ones(mkpts1.shape[0]).view(1, -1, 1),
                ),
                torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
                K.tensor_to_image(img1),
                K.tensor_to_image(img2),
                inliers,
                draw_dict={
                    "inlier_color": (0.2, 1, 0.2),
                    "tentative_color": (1.0, 0.5, 1),
                    "feature_color": (0.2, 0.5, 1),
                    "vertical": False,
                },
            )

        inliers = inliers.squeeze()

        return mkpts0[inliers].astype(int), mkpts1[inliers].astype(int)

    def l1_norm(self, p1, p2):
        return np.sum(np.abs(np.array(p1) - np.array(p2)))

    def find_matching_keypoints_for_third_image(self, keypoints12, keypoints13, keypoints23, order=(0,1,2), max_dist=2):
        o1, o2, o3 = order
        matched_keypoints = [[] for _ in range(3)]
        tree1 = spatial.KDTree(keypoints13[0])
        tree2 = spatial.KDTree(keypoints23[0])
        queried_dist1, queried_closest_inds1 = tree1.query(keypoints12[0], k=1, distance_upper_bound=max_dist)
        queried_dist2, queried_closest_inds2 = tree2.query(keypoints12[1], k=1, distance_upper_bound=max_dist)

        for i in range(len(keypoints12[0])):
            dist1, closest_index1 = queried_dist1[i], queried_closest_inds1[i]
            dist2, closest_index2 = queried_dist2[i], queried_closest_inds2[i]
            if dist1 > max_dist and dist2 > max_dist:
                continue
            matched_keypoints[o1].append(keypoints12[0][i])
            matched_keypoints[o2].append(keypoints12[1][i])
            if dist1 < dist2:
                matched_keypoints[o3].append(keypoints13[1][closest_index1])
            else:
                matched_keypoints[o3].append(keypoints23[1][closest_index2])
        
        matched_keypoints = np.array(matched_keypoints)

        return [matched_keypoints[:,i,:].tolist() for i in range(matched_keypoints.shape[1])]

    def find_all_matching_keypoints(self, keypoints12, keypoints13, keypoints23, max_dist=2):
        all_matching_keypoints = []
        all_matching_keypoints += self.find_matching_keypoints_for_third_image(keypoints12, keypoints13, keypoints23, max_dist=max_dist)
        all_matching_keypoints += self.find_matching_keypoints_for_third_image(keypoints13, keypoints12, (keypoints23[1],keypoints23[0]), order=(0,2,1), max_dist=max_dist)
        all_matching_keypoints += self.find_matching_keypoints_for_third_image(keypoints23, (keypoints12[1],keypoints12[0]), (keypoints13[1],keypoints13[0]), order=(1,2,0), max_dist=max_dist)
        return np.unique(np.array(all_matching_keypoints), axis=0)

    def get_angles_in_rad(self, elevation, azimuth):
        # Convert angles to radians
        elevation_rad = np.radians(elevation)
        azimuth_rad = np.radians(azimuth)
        return elevation_rad, azimuth_rad

    def get_xyz(self, elevation1, azimuth1, radius1, keypoints1, elevation2, azimuth2, radius2, keypoints2):
        """
        Taken from https://stackoverflow.com/questions/55740284/how-to-triangulate-a-point-in-3d-space-given-coordinate-points-in-2-image-and-e
        """
        # Get intrinsic matrices
        M = compute_intrinsics(self.fov, self.img_size)

        elevation1_rad, azimuth1_rad = self.get_angles_in_rad(elevation1, azimuth1)
        camera1_A = compute_extrinsics(elevation1_rad, azimuth1_rad, radius1)
        camera1_P = np.dot(M, camera1_A)
        
        elevation2_rad, azimuth2_rad = self.get_angles_in_rad(elevation2, azimuth2)
        camera2_A = compute_extrinsics(elevation2_rad, azimuth2_rad, radius2)
        camera2_P = np.dot(M, camera2_A)
        
        homog_points = cv2.triangulatePoints(camera1_P, camera2_P, keypoints1.astype(float), keypoints2.astype(float)).transpose()
        euclid_points = cv2.convertPointsFromHomogeneous(homog_points)
        
        return euclid_points.squeeze()
    
    def project_3d_to_image(self, point_3d, elevation3, azimuth3, radius3):

        camera3_M = compute_intrinsics(self.fov, self.img_size)

        elevation3_rad, azimuth3_rad = self.get_angles_in_rad(elevation3, azimuth3)
        camera3_A = compute_extrinsics(elevation3_rad, azimuth3_rad, radius3)
        camera3_P = np.dot(camera3_M, camera3_A)

        point_3d_homo = np.array([*point_3d, 1]).reshape(-1,1)
        camera3_coords = camera3_P.dot(point_3d_homo).squeeze()

        return (camera3_coords[:2] / camera3_coords[2]).astype(int)
    
    def adjust_elevation(self, spherical_coordinates, relative_elevation):
        adjusted_spherical_coordinates = []
        for params in spherical_coordinates:
            adjusted_spherical_coordinates.append((relative_elevation + params[0], params[1], params[2]))
        return adjusted_spherical_coordinates

    def calculate_mean_reprojection_error(self, matching_keypoints, adjusted_spherical_coordinates, plot=False):
        error = 0
        combinations = [(0,1,2),(0,2,1),(1,2,0)]
        projected_points = []
        c0,c1,c2 = 0,1,2
        for c0,c1,c2 in combinations:
            for points in matching_keypoints:
                point_3d = self.get_xyz(
                    *adjusted_spherical_coordinates[c0], points[c0],
                    *adjusted_spherical_coordinates[c1], points[c1]
                )
                projected_point = self.project_3d_to_image(point_3d, *adjusted_spherical_coordinates[c2])
                error += self.l1_norm(points[c2], projected_point)
                projected_points.append(projected_point)
            if plot and (c0,c1,c2) == (0,1,2):
                self.plot_projected_points(plot, matching_keypoints, np.array(projected_points))
        mean_error = error / len(matching_keypoints)
        return mean_error
    
    def calculate_total_error_for_elevation(
            self,
            matched_features,
            estimated_elevation,
            plot=False
        ):  
        adjusted_spherical_coordinates = self.adjust_elevation(self.spherical_coordinates, estimated_elevation)
        errors = []
        for triplet in itertools.combinations(range(len(self.images)), 3):
            i1,i2,i3 = triplet
            # get matched keypoints between each pair in the tripley group
            keypoints12 = matched_features[(i1,i2)]
            keypoints13 = matched_features[(i1,i3)]
            keypoints23 = matched_features[(i2,i3)]

            # get common keypoints between all three images
            matching_keypoints = self.find_all_matching_keypoints(keypoints12, keypoints13, keypoints23, max_dist=np.min(self.img_size)*0.03)
            # random_inds = np.random.choice(range(len(matching_keypoints)), 50)
            # matching_keypoints = matching_keypoints[random_inds]

            # triangulate, project and calculate error
            triplet_spherical_coordinates = (adjusted_spherical_coordinates[i1], adjusted_spherical_coordinates[i2], adjusted_spherical_coordinates[i3])
            # print(f"{triplet}: {triplet_spherical_coordinates}")
            if plot and triplet == (0,1,2) and estimated_elevation == self.actual_elevation:
                # print(triplet_spherical_coordinates)
                reprojection_error = self.calculate_mean_reprojection_error(matching_keypoints, triplet_spherical_coordinates, plot=triplet)
            else:
                reprojection_error = self.calculate_mean_reprojection_error(matching_keypoints, triplet_spherical_coordinates)
            errors.append(reprojection_error)

        return np.sum(errors)
    
    def calculate_total_error_for_azimuth(
            self,
            matched_features,
            estimated_azimuth,
            plot=False
        ):  
        adjusted_spherical_coordinates = self.adjust_azimuth(self.spherical_coordinates, estimated_azimuth)
        errors = []
        for triplet in itertools.combinations(range(len(self.images)), 3):
            i1,i2,i3 = triplet
            # get matched keypoints between each pair in the tripley group
            keypoints12 = matched_features[(i1,i2)]
            keypoints13 = matched_features[(i1,i3)]
            keypoints23 = matched_features[(i2,i3)]

            # get common keypoints between all three images
            matching_keypoints = self.find_all_matching_keypoints(keypoints12, keypoints13, keypoints23, max_dist=np.min(self.img_size)*0.03)

            # triangulate, project and calculate error
            triplet_spherical_coordinates = (adjusted_spherical_coordinates[i1], adjusted_spherical_coordinates[i2], adjusted_spherical_coordinates[i3])
            # print(f"{triplet}: {triplet_spherical_coordinates}")
            if plot and triplet == (0,1,2) and estimated_azimuth == self.actual_azimuth:
                reprojection_error = self.calculate_mean_reprojection_error(matching_keypoints, triplet_spherical_coordinates, plot=triplet)
            else:
                reprojection_error = self.calculate_mean_reprojection_error(matching_keypoints, triplet_spherical_coordinates)
            errors.append(reprojection_error)

        return np.sum(errors)
    
    def plot_projected_points(self, triplet, matched_keypoints, projected_points):
        plt.figure(figsize=(12,5))
        for i in range(3):
            plt.subplot(1,4,i+1, xticks=[], yticks=[])
            plt.imshow(Image.open(self.images[triplet[i]]))
        plt.subplot(1,4,1); plt.title("a")
        plt.subplot(1,4,2); plt.title("b")
        plt.subplot(1,4,3); plt.title("LoFTR match")
        plt.subplot(1,4,4, xticks=[], yticks=[])
        plt.title("Reprojection")
        plt.imshow(Image.open(self.images[triplet[2]]))
        
        n = 10
        random_inds = np.random.choice(range(len(matched_keypoints)), n)
        sampled_matched_keypoints = matched_keypoints[random_inds]
        sampled_projected_points = projected_points[random_inds]

        for i in range(n):
            plt.subplot(1,4,1, xticks=[], yticks=[])
            plt.scatter(*sampled_matched_keypoints[i,0,:])
            plt.subplot(1,4,2, xticks=[], yticks=[])
            plt.scatter(*sampled_matched_keypoints[i,1,:])
            plt.subplot(1,4,3, xticks=[], yticks=[])
            plt.scatter(*sampled_matched_keypoints[i,2,:])
            plt.subplot(1,4,4, xticks=[], yticks=[])
            plt.scatter(*sampled_projected_points[i])

        plt.show()