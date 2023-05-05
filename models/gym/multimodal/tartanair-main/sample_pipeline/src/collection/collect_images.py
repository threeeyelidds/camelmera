# collect images in cv mode
import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(_CURRENT_PATH, '..'))
# print(sys.path)

from airsim.types import Pose, Vector3r , Quaternionr
from .ImageClient import ImageClient
from .utils import depth_float32_rgba
from .datatypes.Scene import SceneType
from .datatypes.Depth import DepthType
from .datatypes.Segmentation import SegmentationType
from .datatypes.CubeScene import CubeSceneType
from .datatypes.CubeDistance import CubeDistanceType

import cv2 # debug
import numpy as np
import time

from os import mkdir, listdir
from os.path import isdir, join, isfile
from settings import get_args
import yaml

np.set_printoptions(precision=3, suppress=True, threshold=10000)


class DataSampler(object):
    def __init__(self, data_dir, imgtypes, camlist, disable_cube_cuda = False):

        # self.args = args
        self.datadir = data_dir

        self.imgtypelist = imgtypes.split('_') # Scene_DepthPlanar_Segmentation
        self.camlist = camlist.split('_')

        # this is hard coded, the camlist should only contains cam from 0 to 11 
        self.camlist_name = {'0': 'lcam_front', '1': 'lcam_back', '2': 'lcam_right', '3': 'lcam_left', '4': 'lcam_top', '5': 'lcam_bottom',
                             '6': 'rcam_front', '7': 'rcam_back', '8': 'rcam_right', '9': 'rcam_left', '10': 'rcam_top', '11': 'rcam_bottom'} 
        # the output folder prefix for each modality
        self.imgtype_foldername = {
            'Scene': 'image', 
            'DepthPlanar': 'depth', 
            'Segmentation': 'seg',
            'CubeScene': 'cube_image',
            'CubeDistance': 'cube_dist'}
        # the class handle the data save for each modality
        self.type_converter = {
            'Scene': SceneType, 
            'DepthPlanar': DepthType, 
            'Segmentation': SegmentationType,
            'CubeScene': CubeSceneType,
            'CubeDistance': CubeDistanceType}

        self.imgclient = ImageClient(self.camlist, self.imgtypelist)
        self.imgclient.cube_cuda = not disable_cube_cuda

        self.logfile = self.datadir + '/sample.log'

        # store all the data modalities according to camlist and imgtypelist
        # this is a dictionary, key: TYPE_CAM, value: object with class under the datatypes folder
        self.dataclass_dict = {} 

        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files

    def save_config(self, trajfolder):
        # The config file is saved to the trajectory folder
        config = {'image_type': self.imgtypelist, 
                    'camera_list': self.camlist,
                    'camlist_name': self.camlist_name,
                    'type_name': self.imgtype_foldername}
        collection_config_file = join(self.datadir, trajfolder, 'collection_config.yaml')
        if not isfile(collection_config_file):
            with open(collection_config_file, 'w') as f:
                yaml.dump(config, f)
        else:
            print("Config file exists: {}".format(collection_config_file))

    def create_folders(self, trajdir):
        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files
        for camind in self.camlist:
            camname = self.camlist_name[camind]
            for imgtype in self.imgtypelist:
                assert imgtype in self.imgtype_foldername, "Imgtype not supported {}".format(imgtype)
                datakey = imgtype+'_'+camind
                folderdir = join(trajdir, self.imgtype_foldername[imgtype] + '_' + camname)
                if not isdir(folderdir):
                    mkdir(folderdir)

                # initialize the type converters
                assert imgtype in self.type_converter, "Imgtype not supported {}".format(imgtype)
                typeclass = self.type_converter[imgtype]
                self.dataclass_dict[datakey] = typeclass(folderdir, camname)
            # create pose file
            self.posefilelist.append(trajdir+'/pose_'+camname+'.txt')
            self.posenplist.append([])

    def init_folders(self, traj_folder, disable_timestr=False):
        '''
        traj_folder: string that denotes the folder name, e.g. T000
        '''
        if not isdir(self.datadir):
            mkdir(self.datadir)
        else: 
            print ('Data folder already exists.. {}'.format(self.datadir))

        print('Data output to {}'.format(self.datadir))

        trajdir = join(self.datadir, traj_folder)
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        if not isdir(trajdir):
            mkdir(trajdir)
            self.create_folders(trajdir)
        elif disable_timestr:
            print ('Trajectory folder already exists! {}, but forced to overwrite the data.'.format(trajdir))
            self.create_folders(trajdir)
        else:
            traj_folder =  traj_folder + '_' + timestr
            print ('Trajectory folder already exists! {}, create folder with time stamp {}.'.format(trajdir, traj_folder))
            trajdir = join(self.datadir, traj_folder)
            mkdir(trajdir)
            self.create_folders(trajdir)
        return traj_folder

    def data_sampling(self, poses, trajname, save_data=True, MAXTRY=3, disable_timestr=False, 
                        dyna_time=0, max_framenum=0): 
        '''
        poses: N x 7 numpy array
        trajname: trajectory folder name e.g. Pxxx
        save_data: if not save_data, open posefiles will be stored
        '''

        trajname = self.init_folders(trajname, disable_timestr)

        # save a config file in the output folder
        self.save_config(trajname)

        with open(self.logfile,'a') as f:
            f.write('Sample trajname '+ trajname+'\n')

        time.sleep(5.0)
        data_dict, camposelist = self.imgclient.readimgs() # just do a first query, because the first frame usually has issues
        time.sleep(5.0)
        data_dict, camposelist = self.imgclient.readimgs() # just do a first query, because the first frame usually has issues
        
        framenum = len(poses)
        if max_framenum!=0:
            framenum = min(framenum, max_framenum)
            poses = poses[0:framenum, :]
        print('Sample trajname {} with {} frames..'.format(trajname, framenum))

        start = time.time()
        for k, pose in enumerate(poses):
            position = Vector3r(pose[0], pose[1], pose[2])
            orientation = Quaternionr(pose[3], pose[4], pose[5], pose[6])

            dronepose = Pose(position, orientation)
            self.imgclient.setpose(dronepose)

            if save_data:
                if dyna_time > 0:
                    self.imgclient.simPause(False)
                    time.sleep(dyna_time) # image rate
                self.imgclient.simPause(True)
                data_dict, camposelist = self.imgclient.readimgs()
                # handle read image error
                if data_dict is None:
                    # try read the images again
                    print ('  !!Error read image: {}-{}: {}'.format(trajname, k, pose))
                    for s in range(MAXTRY):
                        time.sleep(0.01)
                        data_dict, camposelist = self.imgclient.readimgs()
                        if data_dict is not None:
                            break
                        else:
                            print ('    !!Error retry read image: Retry {}'.format(s))
                if dyna_time == 0: # for static envs, just let the simulation go while saving the data
                    self.imgclient.simPause(False)
            else:
                camposelist = []
                for camind in self.camlist:
                    camposelist.append([pose[0], pose[1], pose[2],orientation.x_val,orientation.y_val,orientation.z_val,orientation.w_val])

            if data_dict is None:
                print ('  !!Can not recover from read image error {}'.format(trajname))
                return

            # save images and poses
            if save_data:
                dataindex = str(k).zfill(6)
                for datatype in self.dataclass_dict:
                    self.dataclass_dict[datatype].save_file(data_dict[datatype], dataindex)

            for w,camind in enumerate(self.camlist):
                # write pose to file
                self.posenplist[w].append(np.array(camposelist[w]))

                # imgshow = np.concatenate((leftimg,rightimg),axis=1)
            print ('  {0}, pose {1}, \torientation ({2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f})'.format(k, pose[:3], orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val))
            # cv2.imshow('img',imgshow)
            # cv2.waitKey(1)

        for w in range(len(self.camlist)):
            # save poses into numpy txt
            np.savetxt(self.posefilelist[w], self.posenplist[w])

        end = time.time()
        print('Trajectory sample time {}'.format(end - start))

        with open(self.logfile,'a') as f:
            f.write('    Success! Traj len: {}, time {} min. \n'.format(len(poses), (end-start)/60.0))

    def close(self,):
        self.imgclient.close()


def collect_data_files(pose_folder, output_dir, imgtypes, camlist, disable_cube_cuda = False, 
                        save_posefile_only = False, dyna_time = 0, 
                        traj_overwrite = False, max_framenum = 0):
    '''
    Enumerate the Pose_xxx.txt in the pose_folder
    Collect trajectory data and put them in the output_dir
    1. args.load_existing_trajectories = False: collect data from single pose file that generated from trajectory sampling
    2. args.load_existing_trajectories = True:  collect data from pose file in already existed trajecotry folder
    Example: python -m collection.collect_images \
                --environment-dir /home/amigo/tmp/test_sample_trajs \
                --posefile-folder /home/amigo/tmp/test_sample_trajs/pose_test \
                --data-folder Data \
                --cam-list 0_1_2_3_4_5
    '''
    datasampler = DataSampler(output_dir, imgtypes, camlist, disable_cube_cuda)

    posfiles = listdir(pose_folder)
    posfiles = [ff for ff in posfiles if (ff[-3:]=='txt' and ff[:4]=='Pose')]
    posfiles.sort()

    for posfilename in posfiles:
        outfoldername = posfilename.split('.txt')[0].replace('Pose_', 'P')
        print ('*** {} ***'.format(outfoldername))

        poses = np.loadtxt(join(pose_folder, posfilename))
        datasampler.data_sampling(poses, outfoldername, save_data=(not save_posefile_only), disable_timestr=traj_overwrite,
                                    dyna_time=dyna_time, max_framenum=max_framenum)

    datasampler.close()

def collect_data_from_existing_trajecotry_folder(traj_folder, output_dir, imgtypes, camlist, 
                                                    disable_cube_cuda = False, posefilename = 'pose_lcam_front.txt', 
                                                    save_posefile_only = False, dyna_time = 0, traj_overwrite = False, max_framenum=0):
    '''
    Used for resample images along the same trajectory according to the pose files such as pose_left.txt and pose_right.txt
    args.load_existing_trajectories = True 
    traj_folder: the input folder, usually: root/env/data/
    posefilename: the pose file used to set the camera poses
    Example: python -m collection.collect_images \
                --load-existing-trajectories \
                --posefile-name pose_lcam_front.txt \
                --environment-dir /home/amigo/tmp/test_sample_downtown2_new \
                --traj-folder /home/amigo/tmp/test_sample_downtown2/Data \
                --data-folder Data \
                --cam-list 0_1_2_3_4_5
    '''
    datasampler = DataSampler(output_dir, imgtypes, camlist, disable_cube_cuda)

    subfolders = listdir(traj_folder)
    subfolders = [ff for ff in subfolders if ff[0]=='P']
    subfolders.sort()

    for subfolder in subfolders:
        print ('*** {} ***'.format(subfolder))
        subfolderpath = join(traj_folder, subfolder)
        posefile = join(subfolderpath, posefilename)
        poses = np.loadtxt(posefile)

        datasampler.data_sampling(poses, subfolder, save_data=(not save_posefile_only), disable_timestr=traj_overwrite, 
                                    dyna_time=dyna_time, max_framenum=max_framenum)
            # disable_timestr=True)

    datasampler.close()

if __name__ == '__main__':

    args = get_args()

    env_dir = args.environment_dir # output directory 
    data_folder = args.data_folder # output folder
    output_dir = join(env_dir, data_folder)

    camlist = args.cam_list
    imgtypes = args.img_type
    load_existing_traj = args.load_existing_trajectories
    disable_cube_cuda = args.disable_cube_cuda
    
    dyna_time = args.dyna_time
    traj_overwrite = args.traj_overwrite
    max_framenum = args.max_framenum

    if load_existing_traj:
        traj_folder = args.traj_folder
        posefilename = args.posefile_name
        save_posefile_only = args.save_posefile_only
        collect_data_from_existing_trajecotry_folder(traj_folder, output_dir, imgtypes, camlist, 
                                                disable_cube_cuda, posefilename=posefilename, 
                                                save_posefile_only=save_posefile_only, dyna_time=dyna_time,
                                                traj_overwrite=traj_overwrite, max_framenum=max_framenum)
    else:
        pose_folder = args.posefile_folder
        collect_data_files(pose_folder, output_dir, imgtypes, camlist, disable_cube_cuda, 
                            dyna_time=dyna_time, traj_overwrite=traj_overwrite, max_framenum=max_framenum)

