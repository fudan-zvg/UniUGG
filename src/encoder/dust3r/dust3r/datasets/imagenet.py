# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import os.path as osp
import os
import cv2
import numpy as np
import random

from dust3r.datasets.base.base_nature_img import BaseNatureImg
from dust3r.utils.image import imread_cv2

import glob

class ImageNet(BaseNatureImg):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        # self.loaded_data = self._load_data()
        # self.images = os.listdir(ROOT)
        # self.images1 = self.images[1::2]
        # self.images2 = self.images[0::2]
        self.images = glob.glob(os.path.join(ROOT, '**', '*.JPEG'), recursive=True)
        random.shuffle(self.images)
        self.images = [os.path.relpath(p, ROOT) for p in self.images]

    # def _load_data(self):
    #     with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
    #         self.scenes = data['scenes']
    #         self.sceneids = data['sceneids']
    #         self.images = data['images']
    #         self.intrinsics = data['intrinsics'].astype(np.float32)
    #         self.trajectories = data['trajectories'].astype(np.float32)
    #         self.pairs = data['pairs'][:, :2].astype(int)

    def __len__(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng):

        basename1 = self.images[idx]
        basename2 = random.choice(self.images)

        views = []    
        # Load RGB image
        for basename in [basename1, basename2]:

            rgb_image = imread_cv2(osp.join(self.ROOT, basename))
            # Load depthmap
            rgb_image = self._crop_resize_if_necessary(rgb_image, resolution, rng=rng)

            views.append(dict(
                img=rgb_image,
                dataset='ImageNet'
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ImageNet(ROOT="data/imagente-1k/train_data_process2", resolution=224, aug_crop=16)
    print(len(dataset))
    print(len(dataset[0]))
    print(dataset.images[0])
    # for idx in np.random.permutation(len(dataset)):
    #     views = dataset[idx]
    #     assert len(views) == 2
    #     print(view_name(views[0]), view_name(views[1]))
    #     viz = SceneViz()
    #     poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
    #     cam_size = max(auto_cam_size(poses), 0.001)
    #     for view_idx in [0, 1]:
    #         pts3d = views[view_idx]['pts3d']
    #         valid_mask = views[view_idx]['valid_mask']
    #         colors = rgb(views[view_idx]['img'])
    #         viz.add_pointcloud(pts3d, colors, valid_mask)
    #         viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
    #                        focal=views[view_idx]['camera_intrinsics'][0, 0],
    #                        color=(idx*255, (1 - idx)*255, 0),
    #                        image=colors,
    #                        cam_size=cam_size)
    #     viz.show()
