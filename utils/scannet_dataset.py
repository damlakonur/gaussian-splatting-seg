from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path
from copy import deepcopy
import json

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset

# Suppress Literal import error for Python < 3.8
try:
    from typing import Literal
except ImportError:
    from typing import Union
    Literal = Union  # Fallback for Python < 3.8

from utils.scene_release import ScannetppScene_Release
from utils.colmap import read_model
from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2


class ScannetppDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        scene_id: str,
        split: Literal["train", "test"] = "train",
        preload_images: bool = True,
        max_images: int = -1,
        return_images: bool = True,
        image_subdir: str = None,
        mask_subdir: str = None,
        transform_file: str = None,
    ):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            scene_id (str): Scene ID to load.
            split (str): Split to load ('train' or 'test').
            preload_images (bool): Whether to preload images into memory. This can use a lot of memory for scenes with many images.
            max_images (int): Maximum number of images to load. If -1, all images are loaded. Otherwise, it will uniformly sample max_images frames from the dataset.
            image_subdir (str): Custom image subdirectory (e.g., 'resized_undistorted_images_4x'). If None, uses default.
            mask_subdir (str): Custom mask subdirectory (e.g., 'resized_undistorted_masks_4x'). If None, uses default.
            transform_file (str): Custom transform file (e.g., 'transforms_undistorted_4x.json'). If None, uses default.
        """
        super().__init__()
        self.root_dir = root_dir
        self.scene_id = scene_id
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.split = split
        self.preload_images = preload_images
        self.return_images = return_images
        self.znear = 0.01
        self.zfar = 100.0

        self.scene = ScannetppScene_Release(
            data_root=root_dir,
            scene_id=scene_id,
        )

        self.colmap_dir = self.scene.dslr_colmap_dir
        
        # Use custom paths if provided, otherwise use defaults
        if transform_file:
            self.json_path = self.scene.dslr_dir / 'nerfstudio' / transform_file
        else:
            self.json_path = self.scene.dslr_nerfstudio_transform_undistorted_path
            
        if image_subdir:
            self.images_dir = self.scene.dslr_dir / image_subdir
        else:
            self.images_dir = self.scene.dslr_resized_undistorted_dir
            
        if mask_subdir:
            self.mask_dir = self.scene.dslr_dir / mask_subdir
        else:
            self.mask_dir = self.scene.dslr_resized_undistorted_mask_dir

        with open(self.json_path, "r") as f:
            transforms = json.load(f)

        self.height = int(transforms["h"])
        self.width = int(transforms["w"])
        self.fx = float(transforms["fl_x"])
        self.fy = float(transforms["fl_y"])
        self.cx = float(transforms["cx"])
        self.cy = float(transforms["cy"])
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)

        if self.split == "train":
            frames = transforms["frames"]
        else:
            frames = transforms["test_frames"]

        self.has_mask = transforms.get("has_mask", False)

        # Sort frames by file_path
        frames = sorted(frames, key=lambda x: x["file_path"])

        if self.split == "train" and max_images > 0 and max_images < len(frames):
            # Uniformly sample max_images frames
            sample_indices = np.linspace(0, len(frames) - 1, max_images, dtype=np.int32)
            frames = [frames[idx] for idx in sample_indices]

        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.images = []
        self.masks = []
        for frame in frames:
            # Convert the poses from nerfstudio to colmap convention
            camera_to_world = np.array(frame["transform_matrix"])
            camera_to_world[2, :] *= -1
            camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
            camera_to_world[0:3, 1:3] *= -1

            self.poses.append(camera_to_world.astype(np.float32))
            self.image_paths.append(os.path.join(self.images_dir, frame["file_path"]))
            self.mask_paths.append(os.path.join(self.mask_dir, frame["file_path"].replace(".JPG", ".png")))

        self.is_testing_scene = False
        if self.split == "test":
            for image_path in self.image_paths:
                if not os.path.exists(image_path):
                    print("> Test images not found. This is expected if the scene is ScanNet++ testing scene. No evaluation will be performed during training.")
                    self.is_testing_scene = True
                    break

        if self.preload_images:
            for i in tqdm(range(len(self.image_paths)), desc="Loading images", total=len(self.image_paths)):
                image_path = self.image_paths[i]
                mask_path = self.mask_paths[i]
                self.images.append(self.load_image(image_path))
                self.masks.append(self.load_mask(mask_path))

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from the given path."""
        if self.split != "train" and not os.path.exists(image_path):
            # It's possible that the scene is nvs testing scene and so the testing images are not available
            # Use a dummy image with all pixels black in this case.
            image = np.zeros((3, self.height, self.width), dtype=np.float32)
        else:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image) / 255.0
            image = image.astype(np.float32)
            image = image.transpose(2, 0, 1)
        return image

    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load a mask from the given path. Pixels with 0: invalid, 255: valid."""
        if self.has_mask:
            if self.split != "train" and not os.path.exists(mask_path):
                # It's possible that the scene is nvs testing scene and so the testing images are not available.
                # Use a dummy mask with all pixels valid in this case.
                mask = np.ones((self.height, self.width), dtype=np.float32)
            else:
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask).astype(np.float32)
                mask = np.clip(mask, 0, 1)
                mask = mask.astype(np.float32)
            mask = mask[None, :, :]  # Add a channel dimension
        else:
            mask = None
        return mask

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the point cloud from the COLMAP model."""
        cameras, images, points3D = read_model(self.colmap_dir, ".txt")
        xyz = []
        rgb = []
        for point in points3D.values():
            xyz.append(point.xyz)
            rgb.append(point.rgb)
        xyz = np.array(xyz)
        rgb = np.array(rgb) / 255.0
        return xyz, rgb

    def get_nerfpp_norm(self) -> Dict[str, np.ndarray]:
        camera_centers = np.array([pose[:3, 3] for pose in self.poses])
        avg_camera_center = np.mean(camera_centers, axis=0)
        dists = np.linalg.norm(camera_centers - avg_camera_center[None, :], axis=1)
        radius = np.max(dists) * 1.1
        translate = -avg_camera_center
        return {"radius": radius, "translate": translate}

    def get_image_names(self) -> List[str]:
        """Get the image names from the dataset."""
        image_names = []
        for image_path in self.image_paths:
            image_name = os.path.basename(image_path)
            image_names.append(image_name)
        return image_names

    def _get_3dgs_info(
        self,
        world_to_camera: np.ndarray,
        fov_x: float,
        fov_y: float,
    ):
        # Prepare the transform and projection matrices for 3DGS
        rot = np.transpose(world_to_camera[:3, :3])
        trans = world_to_camera[:3, 3]
        world_view_transform = torch.from_numpy(getWorld2View2(rot, trans)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=fov_x,
            fovY=fov_y,
        ).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        outputs = {}
        outputs["world_view_transform"] = world_view_transform
        outputs["projection_matrix"] = projection_matrix
        outputs["full_proj_transform"] = full_proj_transform
        outputs["camera_center"] = world_view_transform.inverse()[3, :3]
        return outputs

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image_name = os.path.basename(image_path)

        if self.return_images:
            if self.preload_images:
                image = self.images[idx]
                mask = self.masks[idx] if self.has_mask else None
            else:
                image = self.load_image(image_path)
                mask = self.load_mask(mask_path)

            image = torch.from_numpy(image)
            # mask is already [1, H, W] from load_mask, just convert to tensor
            mask = torch.from_numpy(mask) if mask is not None else None

        pose = self.poses[idx]
        world_to_camera = np.linalg.inv(pose)
        outputs = self._get_3dgs_info(
            world_to_camera=world_to_camera,
            fov_x=self.fovx,
            fov_y=self.fovy,
        )
        intrinsic = torch.from_numpy(self.K).float()

        if self.return_images:
            outputs.update({
                "image": image,
                "mask": mask,
            })

        outputs.update({
            "camera_to_world": pose,
            "image_path": image_path,
            "mask_path": mask_path if self.has_mask else None,
            "image_width": self.width,
            "image_height": self.height,
            "intrinsic": intrinsic,
            "fovx": self.fovx,
            "fovy": self.fovy,
            "znear": self.znear,
            "zfar": self.zfar,
            "image_name": image_name,
        })

        return outputs

    def collate_fn(self, batch):
        # Directly return list of dictionaries
        return batch
