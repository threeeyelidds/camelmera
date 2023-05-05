from os.path import isfile, join, isdir
import numpy as np
from .transformation import pose_quats2motion_ses
from .data_enumeration import enumerate_trajs

posefilelist_v2 = [
    'pose_lcam_back.txt',
    'pose_lcam_bottom.txt',
    'pose_lcam_front.txt',
    'pose_lcam_left.txt',
    'pose_lcam_right.txt',
    'pose_lcam_top.txt',
    'pose_rcam_back.txt',
    'pose_rcam_bottom.txt',
    'pose_rcam_front.txt',
    'pose_rcam_left.txt',
    'pose_rcam_right.txt',
    'pose_rcam_top.txt',
]

posefilelist_v1 = [
    'pose_left.txt',
    'pose_right.txt'
]

def loadMotionFromPoseFile(trajdir, frame_skip, posefilelist):
    for posefile in posefilelist:
        poses = np.loadtxt(join(trajdir,posefile)).astype(np.float32) # framenum
        cammotions = pose_quats2motion_ses(poses, skip=frame_skip) # framenum - 1 - skip

        print('  Generated {} motion frames from file {}'.format(len(cammotions), trajdir))
    
        motionfilename = posefile.split('.txt')[0].replace('pose', 'motion') + '{}.npy'.format(str(frame_skip+1) if frame_skip>0 else '')
        np.save(join(trajdir, motionfilename), cammotions)

import argparse
def get_args():
    parser = argparse.ArgumentParser(description='TartanAir')
    parser.add_argument('--data-root-dir', default='./', help='root directory for downloaded files')
    parser.add_argument('--tartanair', default='v1', help='v1 or v2')
    args = parser.parse_args()
    return args

def gen_motion(rootdir, datafolders, posefilelist):
    import ipdb;ipdb.set_trace()
    trajdict = enumerate_trajs(rootdir, data_folders = datafolders)
    skiplist = [0,1,3,5]
    for env in trajdict:
        print(env)
        trajlist = trajdict[env]
        for trajdir in trajlist:
            print(trajdir)
            for skip in skiplist:
                loadMotionFromPoseFile(join(data_root_dir, env, trajdir), skip, posefilelist)

# python -m postprocessing.motionfile_generate --data-root-dir /home/amigo/tmp/data/tartan --tartanair v1
if __name__ == '__main__':
    args = get_args()
    data_root_dir = args.data_root_dir
    if args.tartanair == 'v1':
        gen_motion(data_root_dir, datafolders=['Data','Data_fast'], posefilelist=posefilelist_v1)
    elif args.tartanair == 'v2':
        gen_motion(data_root_dir, datafolders=['Data_easy','Data_hard'], posefilelist=posefilelist_v2)
