"""
Author: Yorai Shaoul using base code from Yaoyu Hu
    yorai@cmu.edu
    yaoyuh@andrew.cmu.ecu
    2022-Oct-14
"""

'''
Example execution:

python fish_and_pano.py  --data-root D:\TartanAir_v2  --env-folders ApocalypticCity,Cyberpunk  --data-folders Data_easy,Data_hard  --modalities image,depth,seg  --new-cam-models fish,equirect  --np 12
'''
import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

from settings import get_args
# Python imports.
# import argparse
import multiprocessing
from multiprocessing.util import LOGGER_NAME
import os
from pydoc import ModuleScanner
import sys
import shutil
import time
import cv2
from functools import partial
import numpy as np
from tqdm import tqdm
import yaml
from os.path import split

from multiprocessing import Pool

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _TOP_DIR)

# Visualization.
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# from data_collection.mvs_utils.point_cloud_helper import write_PLY
# import torch
from data_collection.image_sampler.blend_function import BlendBy2ndOrderGradTorch

# Multiview stereo imports.
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import LinearSphere, Equirectangular
from data_collection.image_sampler import SixPlanarTorch
from data_collection.image_sampler.six_images_common import (
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM )
from data_collection.mvs_utils.ftensor import f_eye

# Multiprocessing with logging.
from .process_pool import PoolWithLogger
from os.path import join, isdir
from os import mkdir
# Data enumeration.
from .data_enumeration import enumerate_trajs, enumerate_frames
from .data_visualization import DataVisualizer

# def write_distance_image_as_ply( out_fn, camera_model, dist_img, max_dist=100 ):
#     pixel_coordinates = camera_model.pixel_coordinates() # 2xN.
#     pixel_rays, ray_valid_mask = camera_model.pixel_2_ray( pixel_coordinates )
    
#     if isinstance(pixel_rays, torch.Tensor):
#         pixel_rays = pixel_rays.cpu().numpy()
#         ray_mask = ray_valid_mask.cpu().numpy()
    
#     points = pixel_rays * dist_img.reshape((-1,)) # Broadcast automatically.
    
#     mask = np.linalg.norm(points, axis=0) < max_dist
#     mask = np.logical_and(mask, ray_mask)
    
#     write_PLY(out_fn, points[:, mask])

class TartanAirV2FishPanoPostProcessor():
    def __init__(self) -> None:
        args = get_args()

        # Member variables.
        # The root directory for the dataset.
        self.data_root = args.data_root

        # Individual environment directories within the root directory to be post-processed. If empty, all the available directories will be processed.
        if not args.env_folders:
            self.env_folders = []
        else:
            self.env_folders = args.env_folders.split(",")
            
        # The data-folders to be processed within the environments.
        if not args.data_folders:
            self.data_folders = []
        else:
            self.data_folders = args.data_folders.split(",")

        if not args.traj_folders:
            self.traj_folders = []
        else:
            self.traj_folders = args.traj_folders.split(',')
            if  self.traj_folders[0]=='': # trajectories are specified by the user
                self.traj_folders = []

        # Get configuration parameters from a yaml file.
        with open(os.path.join(os.path.dirname(__file__),'config.yaml'), 'r') as file:
           postproc_config = yaml.safe_load(file)
           self.fish_params = postproc_config['fisheye']
           self.equirect_params = postproc_config['equirect']

        # The modalities to be postprocessed within the environment data.
        if not args.modalities:
            self.modalities = []
        else:
            self.modalities = args.modalities.split(",")

        # The camera models to be generate.
        if not args.new_cam_models:
            self.new_cam_models = []
        else:
            self.new_cam_models = args.new_cam_models.split(",")

        # Number of processes to used.
        self.num_workers = args.np

        # Store the matrix used for depth-to-distance calcluation
        self.conv_matrix = None
        self.depth_shape = None

    # def get_args(self):
    #     parser = argparse.ArgumentParser()
    #     """
    #     Parse arguments from the command line call.
    #     """

    #     parser.add_argument('--data-root', type=str, default="D:/TartanAir_V2",
    #                         help='Path to the root dataset directory. (default: "D:/TartanAir_V2")')

    #     parser.add_argument('--env-folders', type=str, default="",
    #             help='Comma-separated list of environment folders within the root folder. (default: "")')
           
    #     parser.add_argument('--data-folders', type=str, default="Data_easy,Data_hard",
    #             help='Comma-separated list of data folders within the env folders. (default: "Data_easy,Data_hard")')
        
    #     parser.add_argument('--modalities', type=str, default="image,depth,seg",
    #             help='Comma-separated list of input modalities to postprocess. (default: "image,depth,seg")')
        
    #     parser.add_argument('--new-cam-models', type=str, default="fish,equirect",
    #         help='Comma-separated list of camera models to generate. (default: "fish,equirect")')
        
    #     parser.add_argument('--np', type=int, default=1,
    #         help='Integer number of processes to spin up. (default: 1)')
        
        
    #     # Print the values of the args.
    #     return parser.parse_args()

    def depth_to_dist(self, depth):
        '''
        assume: fov = 90 on both x and y axes, and optical center is at image center.
        '''
        # import ipdb;ipdb.set_trace()
        if self.depth_shape is None or \
            depth.shape != self.depth_shape or \
            self.conv_matrix is None: # only calculate once if the depth shape has not changed
            hh, ww = depth.shape
            f = ww/2
            wIdx = np.linspace(0, ww - 1, ww, dtype=np.float32) + 0.5 - ww/2 # put the optical center at the middle of the image
            hIdx = np.linspace(0, hh - 1, hh, dtype=np.float32) + 0.5 - hh/2 # put the optical center at the middle of the image
            u, v = np.meshgrid(wIdx, hIdx)
            dd = np.sqrt(u * u + v * v + f * f)/f
            self.conv_matrix = dd
        self.depth_shape = depth.shape
        disp = self.conv_matrix * depth
        return disp

    def ocv_read(self, fn ):
        image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert image is not None, \
            f'{fn} read error. '
        return image

    def read_rgb(self, fn ):
        return self.ocv_read(fn)

    def read_dist(self, fn ): # read a depth image and convert it to distance
        depth = self.read_dep(fn)
        return self.depth_to_dist(depth)

    def read_dep(self, fn ):
        image = self.ocv_read(fn)
        depth = np.squeeze( image.view('<f4'), axis=-1 )
        return depth

    def vis_dep(self, fn):
        depth = self.read_dep(fn)
        depthvis = np.clip(1/depth * 400, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)

    def read_seg(self, fn ):
        image = self.ocv_read(fn)
        return image.astype(np.uint8)

    def ocv_write(self, fn, image ):
        cv2.imwrite(fn, image)
    
    def write_as_is(self, fn, image ):
        self.ocv_write( fn, image )
    
    def write_float_compressed(self, fn, image):
        assert(image.ndim == 2), 'image.ndim = {}'.format(image.ndim)
    
        # Check if the input array is contiguous.
        if ( not image.flags['C_CONTIGUOUS'] ):
            image = np.ascontiguousarray(image)

        dummy = np.expand_dims( image, 2 )
        self.ocv_write( fn, dummy )
        
    def write_float_depth(self, fn, image):
        if len(image.shape) == 2:
            image = image[...,np.newaxis]
        depth_rgba = image.view("<u1")
        self.ocv_write(fn, depth_rgba)

    def read_images_tartanair_v2(self, dir_name, reader, frame_ix, modality, cam_name):
        '''
        Read the input images based on the hardcoded names.
        
        dir_name (str): The input directory.
        reader (callable): The funciton for reading the images.
        frame_ix (str): the number of the frame as it appears in the file name.
        modality (str): the modality of the image to be processed. E.g., 'depth', 'seg', 'image'. 
        cam_name (str): the camera type to be produces. E.g., 'fish', or 'equirect'.

        Returns:
        A dictionary of all the six images.
        '''
        global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
        
        input_names = {
            FRONT:  os.path.join( dir_name, "_".join([modality, cam_name, "front"]), f'{"_".join([frame_ix, cam_name, "front", modality])}.png' ),
            BACK:  os.path.join( dir_name, "_".join([modality, cam_name, "back"]), f'{"_".join([frame_ix, cam_name, "back", modality])}.png' ),
            LEFT:  os.path.join( dir_name, "_".join([modality, cam_name, "left"]), f'{"_".join([frame_ix, cam_name, "left", modality])}.png' ),
            RIGHT:  os.path.join( dir_name, "_".join([modality, cam_name, "right"]), f'{"_".join([frame_ix, cam_name, "right", modality])}.png' ),
            TOP:  os.path.join( dir_name, "_".join([modality, cam_name, "top"]), f'{"_".join([frame_ix, cam_name, "top", modality])}.png' ),
            BOTTOM:  os.path.join( dir_name, "_".join([modality, cam_name, "bottom"]), f'{"_".join([frame_ix, cam_name, "bottom", modality])}.png' )}
        
        # If the modality is 'image', then remove this name from the file name.
        if modality == 'image':
            for cam_key, path in input_names.items():
                modality_str_ix1 =  (path.find(modality))
                modality_str_ix2 = (path.find(modality, modality_str_ix1 +1))
                new_path = path[:modality_str_ix2-1] + path[modality_str_ix2 + len(modality):]
                input_names[cam_key] = new_path
        # Read all the images.
        images = dict()
        for key, value in input_names.items():
            images[key] = reader(value)
            
        return images

    # Multiprocessing.
    def proc_init(self, logger_name, log_queue):
        global PROC_LOGGER
        # The logger.
        PROC_LOGGER = PoolWithLogger.job_prepare_logger(logger_name, log_queue)
        
    def postprocess_tartanair_v2(self):
        """
        This method goes through all of the sequences created for TartanAirV2, and post-processess those that were selected (all if none selected via flags). For each frame, this method creates (at most, could create less if not all modalities or all camera names were specified):
        1. Fisheye and equirectangular images from the left images.  In RGB, depth, and semantic segmentation. 
        2. Fisheye and equirectangular images from the right images. In RGB, depth, and semantic segmentation.
        """
        # The fisheye camera model.
        camera_model_fisheye = LinearSphere(
            fov_degree = self.fish_params['fov_degree'],
            shape_struct=ShapeStruct(H=self.fish_params['height'], W=self.fish_params['width']),
            in_to_tensor=True, 
            out_to_numpy=False)
        
        # The equirectangular mdoel.
        camera_model_equirect = Equirectangular(
            ShapeStruct(H=self.equirect_params['height'], W= self.equirect_params['width']),
            longitude_span=(-np.pi * 3 /2, np.pi/2), 
            latitude_span=(-np.pi/2, np.pi/2),
            open_span=False,
            in_to_tensor=True, 
            out_to_numpy=False)
        
        # The path to the directory that has been populated with TartanAir data. Immediately in this directory are environment-named directories.
        tartanair_path = self.data_root
        envs_to_trajs = enumerate_trajs(tartanair_path, self.data_folders)

        # Visualizer for semantic segmentation.
        # sv = SegmentationVisualizer( os.path.join( tartanair_path, 'seg_rgbs.txt' ) ) #TODO(yoraish): where is this file? The current path is wrong!

        # Some mappings between attributes and parameters.
        modality_to_reader = {"image": self.read_rgb, "depth": self.read_dist, "seg": self.read_seg}
        new_cam_model_to_camera_model = {"fish": camera_model_fisheye, "equirect": camera_model_equirect}
        modality_to_interpolation = {"image": "linear", "seg": "nearest", "depth": "blend"}
        modality_to_writer = {"image": self.write_as_is, "seg": self.write_as_is, "depth": self.write_float_depth}
        
        print("Preparing for fisheye and panorama post-processing.")
        for env_name, env_trajs  in envs_to_trajs.items():
            
            if len(self.traj_folders)>0:
                env_trajs = [tt for tt in env_trajs if split(tt)[-1] in self.traj_folders]

            if self.env_folders and env_name not in self.env_folders:
                print("    Skipping environment", env_name)
                continue
            print("    Environment:", env_name)
            for rel_traj_path in env_trajs: 
                print(os.path.join(tartanair_path, env_name, rel_traj_path))
                traj_path = os.path.join(tartanair_path, env_name, rel_traj_path)

                # For this trajectory folder, create the appropriate folders for each new data input and populate those with resampled images.
                for modality in self.modalities:
                    for cam_name in ['lcam', 'rcam']: 
                        for new_cam_model in self.new_cam_models:
                            # Create directory.
                            new_data_dir_path = os.path.join(tartanair_path, env_name, rel_traj_path, "_".join([modality, cam_name, new_cam_model]))
                            # print("Creating directory", new_data_dir_path)
                            
                            # Does not overwrite older directories if those exist.
                            if os.path.exists(new_data_dir_path):
                                pass
                                # print("    !! New data directory already exists.")
                            else:
                                os.makedirs(new_data_dir_path)

                # For each timestep in the trajectory, collect images and create new-camera-model images.
                # Choose a random modality directory to deternmine the number of frames available.
                random_folder_name = "image_lcam_bottom"
                frames = enumerate_frames(os.path.join(traj_path, random_folder_name), '.png')
                num_frames = len(frames)
                print("        Found", num_frames, "frames.")

                frame_ixs = [f.split("_")[0] for f in frames]
                print("        Preparing frames for multiprocessing.")

                # Keep a list of arguments to be multiprocess-passed to the function `sample_image`.
                sample_image_worker_args = [] 
                count = 0
                starttime = time.time()
                for frame_ix in frame_ixs: 
                    for new_cam_model in self.new_cam_models:
                        # Aggregate the information needed to create a new cam-model image from all the images in this time-step.                    
                        sample_args = [] 

                        for modality in self.modalities:
                            for cam_name in ['lcam', 'rcam']: 
                                sample_image_worker_args.append([traj_path, \
                                                                frame_ix, \
                                                                new_cam_model, \
                                                                modality, \
                                                                cam_name, \
                                                                modality_to_reader, \
                                                                new_cam_model_to_camera_model, \
                                                                modality_to_interpolation, \
                                                                modality_to_writer])
                # Run in parallel.
                try:
                    workernum = min(self.num_workers, 2) # use no more than 4 because the sampling process already uses multi-process
                    with PoolWithLogger(workernum, self.proc_init, 'tartanair', ".\log.txt" ) as pool:
                        results = pool.map( self.sample_image_worker, sample_image_worker_args )
                except KeyboardInterrupt:
                    print("Caught KeyboardInterrupt, terminating workers.")
                    pool.terminate()

                print("Finish sample traj {} in {}s".format(traj_path, time.time()-starttime))
            # Or run sequentially.
            # for arglist in sample_image_worker_args:
            #     self.sample_image_worker(*arglist)

    def sample_image_worker(self, traj_path, frame_ix, new_cam_model, modality, cam_name, modality_to_reader, new_cam_model_to_camera_model, modality_to_interpolation, modality_to_writer): 
        """Worker function to participate in parallelized implementation of postprocessing.

        Args:
            traj_path (str): path to the trajectory folder
            frame_ix (str): frame number as it shows in the file name. Could have prefix/trailing zeros.
            new_cam_model (str): images with this camera model will be created. 
            modality (str): the modality of the input image to be postprocess.
            cam_name (str): which camera the input image is taken from.
            modality_to_reader (dict): mapping between input modality to callable reader functions.
            new_cam_model_to_camera_model (dict): mapping between cam model string to camera model object.
            modality_to_interpolation (dict): mapping between input modality to the interpolation arg.
            modality_to_writer (dict): mapping from input modality to the writer function to be used to write the image.
        """

        global PROC_LOGGER
        PROC_LOGGER.info(f'Processing {traj_path}: {frame_ix}, {new_cam_model}, {modality}, {cam_name}')
        # traj_path, frame_ix, new_cam_model, modality, cam_name, modality_to_reader, new_cam_model_to_camera_model, modality_to_interpolation, modality_to_writer = args
        # Populate with images.
        # print(f"Sampling frame {frame_ix} of {traj_path}")
        output_dir_path = os.path.join( traj_path, "_".join([modality, cam_name, new_cam_model]))#, "_".join([frame_ix, cam_name, new_cam_model, modality]))
        
        image_dict = self.read_images_tartanair_v2(traj_path, modality_to_reader[modality], frame_ix, modality, cam_name)
        sample_args = {
                        'out_dir': output_dir_path,
                        'out_fn_base': "_".join([frame_ix, cam_name, new_cam_model, modality]),
                        'camera_model': new_cam_model_to_camera_model[new_cam_model],
                        'image_dict': image_dict,
                        'interpolation': modality_to_interpolation[modality],
                        'image_writer': modality_to_writer[modality],
                    }

        self.sample_image(**sample_args)

    def sample_image(self, out_dir, out_fn_base, image_writer, camera_model, image_dict, interpolation):
        """The function that samples the images.

        Args:
            out_dir (str): output directory path. Does not include the output file name.
            out_fn_base (str): the output file name. Not a path.
            image_writer (callable): the image writer function to use.
            camera_model (CameraModel): camera model object to use.
            image_dict (dict): information to be used for sampling.
            interpolation (str): interpolation method to be used.
        """
        # print(f'=== Sample fisheye images to {out_dir}. ===')
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # The sampler.
        sampler = SixPlanarTorch(camera_model.fov_degree, camera_model, f_eye(3, f0='cbf', f1='fisheye'))
        # sampler.enable_cuda()
        
        # Sample one image.
        if interpolation == 'blend':
            blend_func = BlendBy2ndOrderGradTorch(0.01) # hard code
            sampled, mask = sampler.blend_interpolation(image_dict, blend_func, invalid_pixel_value=0)  
        else:
            sampled, _ = sampler(image_dict, interpolation=interpolation, invalid_pixel_value=0)
        out_fn = os.path.join(out_dir, '%s.png' % (out_fn_base) )

        # if 'depth' in out_fn_base:
        #     write_distance_image_as_ply(out_fn.replace('.png','.ply'), camera_model, sampled) # for debug
        image_writer(out_fn, sampled)

    def create_video_one_traj(self, args):
        # ----------------
        # |    |    |    | 360  fish-rgb, fish-depth, fish-seg
        # ----------------
        # |     1080     | 540  pano-rgb
        # ----------------
        # Assume both fish and pano exist, fish's three modalities are collected
        # For this trajectory folder, get the appropriate folders for each resampled data type images. 
        env_name, trajdir = args
        fish_rgb_folder = 'image_lcam_fish'
        fish_depth_folder = 'depth_lcam_fish'
        fish_seg_folder = 'seg_lcam_fish'
        pano_rgb_folder = 'image_lcam_equirect'

        vidfishsize = 360 # vis size of the fisheye images
        vidpanosize = 540 # vis height of the equirect images

        fish_rgb_dir = join(self.data_root, env_name, trajdir, fish_rgb_folder)
        fish_depth_dir = join(self.data_root, env_name, trajdir, fish_depth_folder)
        fish_seg_dir = join(self.data_root, env_name, trajdir, fish_seg_folder)
        pano_rgb_dir = join(self.data_root, env_name, trajdir, pano_rgb_folder)

        if not isdir(fish_rgb_dir) or not isdir(fish_depth_dir) or not isdir(fish_seg_dir) or not isdir(pano_rgb_dir):
            print('Missing fish or equirect data folder!')
            return

        viddir = join(self.data_root, env_name, 'video')
        if not isdir(viddir):
            mkdir(viddir)

        visualizer = DataVisualizer()
        video_fpath = join(viddir, trajdir.replace('\\','_').replace('/','_')+'_fish_equirect.mp4')
        vid = cv2.VideoWriter(video_fpath, cv2.VideoWriter_fourcc(*'mp4v'), 10, (vidpanosize*2, vidfishsize+vidpanosize)) 
        print("Saving to video {}".format(video_fpath))
        
        framelist = enumerate_frames(fish_rgb_dir, 'png')
        for frame in framelist: 
            imgfish = self.read_rgb(join(fish_rgb_dir, frame + '_lcam_fish_image.png'))
            depfish = self.read_dep(join(fish_depth_dir, frame + '_lcam_fish_depth.png'))
            depfish = visualizer.visdepth(depfish)
            segfish = self.read_seg(join(fish_seg_dir, frame + '_lcam_fish_seg.png'))
            segfish = visualizer.visseg(segfish)

            imgfish = cv2.resize(imgfish, (vidfishsize,vidfishsize), interpolation=cv2.INTER_NEAREST)
            depfish = cv2.resize(depfish, (vidfishsize,vidfishsize), interpolation=cv2.INTER_NEAREST)
            segfish = cv2.resize(segfish, (vidfishsize,vidfishsize), interpolation=cv2.INTER_NEAREST)

            imgpano = self.read_rgb(join(pano_rgb_dir, frame + '_lcam_equirect_image.png'))
            imgpano = cv2.resize(imgpano, (vidpanosize*2, vidpanosize), interpolation = cv2.INTER_NEAREST)

            vis = np.concatenate((imgfish, depfish, segfish), axis = 1)
            vis = np.concatenate((vis, imgpano), axis = 0)

            vid.write(vis)

        vid.release()

    def create_videos(self):

        # Get all the relevant directories containing post-processed images.
        # The path to the directory that has been populated with TartanAir data. Immediately in this directory are environment-named directories.
        tartanair_path = self.data_root
        envs_to_trajs = enumerate_trajs(tartanair_path, self.data_folders)
        print("Creating a video from post-processed data.")

        params = []
        # import ipdb;ipdb.set_trace()
        for env_name, env_trajs  in envs_to_trajs.items():
            if len(self.traj_folders)>0:
                env_trajs = [tt for tt in env_trajs if split(tt)[-1] in self.traj_folders]

            if self.env_folders and env_name not in self.env_folders:
                print("    Skipping environment", env_name)
                continue
            print("    Environment:", env_name)
            for rel_traj_path in env_trajs: 
                params.append([env_name, rel_traj_path])

        with Pool(self.num_workers) as pool:
            pool.map(self.create_video_one_traj, params)
                                

if __name__ == '__main__':

    startt =time.time()
    postprocessor = TartanAirV2FishPanoPostProcessor()
    args = get_args()
    if not args.fish_video_only:
        postprocessor.postprocess_tartanair_v2()
    postprocessor.create_videos()
    print(f"Process time with {postprocessor.num_workers} workers was {time.time() - startt}")
