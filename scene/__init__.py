#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BasicPointCloud
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], scannetpp_dataset=None):
        """
        Initialize Scene from COLMAP/Blender data or ScanNet++ dataset.
        
        :param args: ModelParams with path configuration
        :param gaussians: GaussianModel instance
        :param load_iteration: Iteration to load from checkpoint
        :param shuffle: Whether to shuffle cameras
        :param resolution_scales: Resolution scales for multi-res training
        :param scannetpp_dataset: Optional ScannetppDataset for ScanNet++ initialization
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # ScanNet++ dataset initialization path
        if scannetpp_dataset is not None:
            self._init_from_scannetpp(args, scannetpp_dataset)
            return

        # Original COLMAP/Blender initialization
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def _init_from_scannetpp(self, args, train_dataset):
        """
        Initialize Scene from ScannetppDataset.
        
        :param args: ModelParams
        :param train_dataset: ScannetppDataset instance (train split)
        """
        os.makedirs(self.model_path, exist_ok=True)
        
        # Get point cloud from dataset's COLMAP data
        xyz, rgb = train_dataset.get_point_cloud()
        normals = np.zeros_like(xyz)
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
        
        # Get camera extent from dataset
        nerf_norm = train_dataset.get_nerfpp_norm()
        self.cameras_extent = nerf_norm["radius"]
        
        # Create fake camera infos for exposure mapping (using image names)
        class FakeCamInfo:
            def __init__(self, image_name):
                self.image_name = image_name
        
        image_names = train_dataset.get_image_names()
        fake_cam_infos = [FakeCamInfo(name) for name in image_names]
        
        # Save input point cloud
        if not self.loaded_iter:
            from plyfile import PlyData, PlyElement
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            elements = np.empty(xyz.shape[0], dtype=dtype)
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            attributes = np.concatenate((xyz, normals, rgb_uint8), axis=1)
            elements[:] = list(map(tuple, attributes))
            vertex_element = PlyElement.describe(elements, 'vertex')
            ply_data = PlyData([vertex_element])
            ply_data.write(os.path.join(self.model_path, "input.ply"))
        
        # Initialize Gaussians from point cloud
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"), 
                                   getattr(args, 'train_test_exp', False))
        else:
            self.gaussians.create_from_pcd(pcd, fake_cam_infos, self.cameras_extent)

    def save(self, iteration, output_path=None):
        """
        Save Gaussians to disk.
        
        :param iteration: Current iteration number
        :param output_path: Optional custom output path (uses self.model_path if None)
        """
        base_path = output_path if output_path is not None else self.model_path
        point_cloud_path = os.path.join(base_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(base_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

