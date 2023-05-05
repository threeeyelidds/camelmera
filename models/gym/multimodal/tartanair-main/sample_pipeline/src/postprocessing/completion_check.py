# check data completion
# read trajectory data file under analyze folder
# go through all the modality folders 
# an option of deep check, where read raw data and make sure they are in the right dimension

import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

import numpy as np
import cv2
from os.path import isfile, join, isdir, split, splitext
from os import listdir, mkdir, environ, system
import time
import yaml
from multiprocessing import Pool
from .data_validation import FileLogger
from .data_enumeration import enumerate_frames

RAW_FOLDER_LIST = [
    'depth_lcam_back',
    'depth_lcam_bottom',
    'depth_lcam_front',
    'depth_lcam_left',
    'depth_lcam_right',
    'depth_lcam_top',
    'depth_rcam_back',
    'depth_rcam_bottom',
    'depth_rcam_front',
    'depth_rcam_left',
    'depth_rcam_right',
    'depth_rcam_top',
    'image_lcam_back',
    'image_lcam_bottom',
    'image_lcam_front',
    'image_lcam_left',
    'image_lcam_right',
    'image_lcam_top',
    'image_rcam_back',
    'image_rcam_bottom',
    'image_rcam_front',
    'image_rcam_left',
    'image_rcam_right',
    'image_rcam_top',
    'seg_lcam_back',
    'seg_lcam_bottom',
    'seg_lcam_front',
    'seg_lcam_left',
    'seg_lcam_right',
    'seg_lcam_top',
    'seg_rcam_back',
    'seg_rcam_bottom',
    'seg_rcam_front',
    'seg_rcam_left',
    'seg_rcam_right',
    'seg_rcam_top',
] 

MOD_FOLDER_LIST =[
    'depth_lcam_equirect',
    'depth_lcam_fish',
    'depth_rcam_equirect',
    'depth_rcam_fish',
    'image_lcam_equirect',
    'image_lcam_fish',
    'image_rcam_equirect',
    'image_rcam_fish',
    'seg_lcam_equirect',
    'seg_lcam_fish',
    'seg_rcam_equirect',
    'seg_rcam_fish',
]

FLOW_FOLDER = 'flow_lcam_front'
IMU_FOLDER = 'imu'
LIDAR_FOLDER = 'lidar'

def parse_datafile( inputfile):
    '''
    trajlist: [TRAJ0, TRAJ1, ...]
    trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
    framelist: [[FRAMESTR0, FRAMESTR1, ...],[FRAMESTR_K, FRAMESTR_K+1, ...], ...]
    '''
    with open(inputfile,'r') as f:
        lines = f.readlines()
    trajlist, trajlenlist, framelist = [], [], []
    ind = 0
    while ind<len(lines):
        line = lines[ind].strip()
        traj, trajlen = line.split(' ')
        trajlen = int(trajlen)
        trajlist.append(traj)
        trajlenlist.append(trajlen)
        ind += 1
        frames = []
        for k in range(trajlen):
            if ind>=len(lines):
                print("Datafile Error: {}, line {}...".format(inputfile, ind))
                raise Exception("Datafile Error: {}, line {}...".format(inputfile, ind))
            line = lines[ind].strip()
            frames.append(line)
            ind += 1
        framelist.append(frames)
    totalframenum = sum(trajlenlist)
    print('{}: Read {} trajectories, including {} frames'.format(inputfile, len(trajlist), totalframenum))
    return trajlist, trajlenlist, framelist, totalframenum

def check_folder_has_files(folderdir, framenum, suffix, logf): 
    trajdir, modfolder = split(folderdir)
    if not isdir(folderdir): 
        logf.logline("  ** {} missing {}".format(trajdir, modfolder))
        return False
    modfiles = enumerate_frames(folderdir, surfix=suffix)

    # remove .az files
    azfiles = [ff for ff in modfiles if ff.startswith('.az')]
    if(len(azfiles)>0):
        cmd = 'rm ' + folderdir + '/.az*'
        system(cmd)
        modfiles = enumerate_frames(folderdir, surfix=suffix)
        print('  -- Removed {} az files in {}'.format(len(azfiles), folderdir))

    if len(modfiles) != framenum:
        logf.logline("  ** {} {} missing frames {}/{}".format(trajdir, modfolder, len(modfiles), framenum))
        return False
    return True

# example: python3 -m postprocessing.completion_check --data-root /ocean/projects/cis220039p/shared/tartanair_v2
if __name__ == '__main__':
    from settings import get_args
    from .data_enumeration import enumerate_trajs

    args = get_args()
    data_root_dir   = args.data_root
    data_folders    = args.data_folders.split(',')
    env_folders     = args.env_folders.split(',')
    traj_folders    = args.traj_folders.split(',')

    
    # Multiprocessing.
    # num_proc        = args.np if args.np > 1 else None
    # deep_check      = args.deep_check

    trajdict = enumerate_trajs(data_root_dir, data_folders)
    if env_folders[0] == '': # env_folders not specified use all envs
        env_folders = list(trajdict.keys())

    logf = FileLogger(join(data_root_dir,'data_completion.log'))

    # import ipdb;ipdb.set_trace()
    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        trajlist = trajdict[env_folder]
        print('Working on env {}'.format(env_dir))
        
        if  traj_folders[0]!='': # trajectories are specified by the user
            trajlist = [tt for tt in trajlist if split(tt)[-1] in traj_folders]

        anainputdir = join(env_dir, 'analyze')
        if not isdir(anainputdir):
            mkdir(anainputdir)

        logf.logline('---')
        logf.logline("{}: traj num {}".format(env_folder, len(trajlist)))
        if len(trajlist) == 0:
            logf.logline("  ** No traj found")
            continue
        
        for trajdir in trajlist:
            logf.log(trajdir + ' ')
        logf.log('\n')

        raw_check_suc = True
        mod_check_suc = True
        for trajdir in trajlist:
            # read the datafile
            traj_dir = join(env_dir, trajdir)
            datastr, trajstr = split(trajdir)
            indexfile_traj = join(anainputdir, 'data_' + env_folder + '_' + datastr+'_'+trajstr+'.txt')
            if not isfile(indexfile_traj):
                logf.logline("  ** Missing datafile {}".format(indexfile_traj))
                raw_check_suc = False
                mod_check_suc = False
                continue
            trajlist, trajlenlist, framelist, totalframenum = parse_datafile(indexfile_traj)
            # check all the modality folders for frame num
            for ff in RAW_FOLDER_LIST:
                modfolder = join(traj_dir, ff)
                raw_check_suc = check_folder_has_files(modfolder, totalframenum, '.png', logf) and raw_check_suc
            
            for ff in MOD_FOLDER_LIST:
                modfolder = join(traj_dir, ff)
                mod_check_suc = check_folder_has_files(modfolder, totalframenum, '.png', logf) and raw_check_suc
            
            flowfolder = join(traj_dir, FLOW_FOLDER)
            mod_check_suc = check_folder_has_files(flowfolder, totalframenum-1, '.png', logf) and raw_check_suc

            lidarfolder = join(traj_dir, LIDAR_FOLDER)
            mod_check_suc = check_folder_has_files(lidarfolder, totalframenum, '.ply', logf) and raw_check_suc

            imu_acc_file = join(traj_dir, IMU_FOLDER, 'acc.npy')
            imu_gyro_file = join(traj_dir, IMU_FOLDER, 'gyro.npy')
            if not isfile(imu_acc_file) or not isfile(imu_gyro_file):
                logf.logline("  ** {} missing imu".format(trajdir))
            else:
                imu_acc = np.load(imu_acc_file)
                imu_gyro = np.load(imu_gyro_file)
                if imu_acc.shape[0] != (totalframenum-1)*10:
                    logf.logline("  ** {} imu-acc missing frames {}/{}".format(trajdir, imu_acc.shape[0], totalframenum))
                    mod_check_suc = False
                if imu_gyro.shape[0] != (totalframenum-1)*10:
                    logf.logline("  ** {} imu-gyro missing frames {}/{}".format(trajdir, imu_gyro.shape[0], totalframenum))
                    mod_check_suc = False

        # write the logfile
        if raw_check_suc:
            logf.logline("  Raw complete".format(trajdir))
        if mod_check_suc:
            logf.logline("  Mod complete".format(trajdir))
        # if deep-check, read the file and check the data size 

        print('==== Finished env {} ===='.format(env_dir))
        print('')
    logf.close()
