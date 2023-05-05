import sys
sys.path.append('..')

import cv2
import glob
import numpy as np
import os
import multiprocessing
import time

# Local packages.
from .SimplePLY import output_to_ply
from .SimulatedLiDARModel import VELODYNE_VLP_16
from .SimulatedLiDAR import convert_DEA_2_Velodyne_XYZ, SimulatedLiDAR
from .ImageReader import ImageReader
from .data_visualization import DataVisualizer


imgreader = ImageReader()
imgvisualizer = DataVisualizer()

isolate_kernel = np.int8([
    [-1, -1, -1],
    [-1, +1, -1],
    [-1, -1, -1],
])

dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

def read_compressed_float(fn, typeStr='<f4'):
    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist. ' % (fn))

    return np.squeeze( 
        cv2.imread(fn, cv2.IMREAD_UNCHANGED).view(typeStr), 
        axis=-1 )


def get_filename_parts(fn):
    p = os.path.split(fn)

    if ( "" == p[0] ):
        p = (".", p[1])

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]


def read_depth_files(root):
    '''
    Arguments: 
    root (string): The root directory for the four directories.

    Returns: 
    A list of list of filenames.
    '''

    # pattern = '%s/**/*_front_*.npy' % (root)
    pattern = '%s/depth_lcam_*/*_front_*.png' % (root)
    frontFns = sorted( glob.glob( pattern, recursive=True ) )

    if ( ( n := len(frontFns) ) == 0 ):
        raise Exception('No files found with %s. ' % (pattern))

    return [ [ fn, fn.replace('front', 'right'), fn.replace('front', 'back'), fn.replace('front', 'left') ] 
            for fn in frontFns ]

def read_depths(fns):
    '''
    Arguments: 
    fns (list of strings): The 4 depth files.

    Returns: 
    A list of depth images.
    '''

    return [read_compressed_float(fn).astype(np.float32) for fn in fns]

def get_depth_change(fn, threshold = 0.1):
    depthnp = imgreader.read_depth(fn)

    gx = cv2.Sobel(depthnp, cv2.CV_32FC1, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(depthnp, cv2.CV_32FC1, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT)
    
    # Sum (norm) the results.
    s = np.sqrt( gx**2 + gy**2 )

    # adaptive threshold wrt the depth value
    adaptive_thresh = np.clip(depthnp.astype(np.float32) * threshold, threshold, 10)

    # Find the over-threshold ones.
    m = s > adaptive_thresh #self.threshold
    m = m.astype(np.float32)

    # m = cv2.erode(m.astype(np.float32), np.ones((2, 2)), borderType=cv2.BORDER_CONSTANT, borderValue=0.0)

    # m = (m>0.5).astype(np.uint8)
    # cv2.imwrite('%s_depth2nd.jpg' % fn.split('/')[-1], m*255)
    # print(m.shape,m.dtype,m.max(),m.min(),m.mean())
    return m

def get_disp_change(fn):
    dispnp = imgreader.read_disparity(fn)
    dispvis = imgvisualizer.visdisparity(dispnp)
    depthchange = cv2.Laplacian(cv2.cvtColor(dispvis, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1)
    depthchange = (depthchange > np.percentile(depthchange, 99)).astype(np.uint8) * 255
    neighbors_all_zero = cv2.morphologyEx(src=depthchange, op=cv2.MORPH_HITMISS, kernel=isolate_kernel)
    depthchange = depthchange & ~neighbors_all_zero
    depthchange = cv2.dilate(depthchange, dilate_kernel, iterations=1)

    # cv2.imwrite('%s_depth2nd.jpg' % fn.split('/')[-1], depthchange)
    
    return (depthchange > 1).astype(int)

def sample_wrapper(args):
    fns, outDir = args
    
    depths = read_depths(fns)

    depthchanges = [get_depth_change(fn) for fn in fns]

    sld = SimulatedLiDAR( 320, 640 )
    sld.set_description( VELODYNE_VLP_16 )
    sld.initialize()

    lidarPoints = sld.extract( depths, depthChangeMask=depthchanges )
    lidarPoints = lidarPoints.reshape( (-1, 3) )
    xyz = convert_DEA_2_Velodyne_XYZ( lidarPoints[:, 0], lidarPoints[:, 1], lidarPoints[:, 2] )

    parts = get_filename_parts( fns[0] )
    outFn = os.path.join( outDir, '%s.ply' % (parts[1].replace('depth', 'lidar')) )

    output_to_ply( outFn, xyz.transpose(), [ 1, xyz.shape[0] ], 100, np.array([0, 0, 0]).reshape((-1,1)), format='binary_little_endian', use_rgb=False )

# python -m postprocessing.lidar 
# --data-root /home/amigo/tmp/test_root 
# --env-folder test_sample_downtown2 
# --data-folders Data_easy

if __name__ == '__main__':
    from settings import get_args
    from .data_enumeration import enumerate_trajs

    args = get_args()

    data_root_dir   = args.data_root
    data_folders    = args.data_folders.split(',')
    env_folders     = args.env_folders.split(',')
    process_num     = args.np

    lidar_folder = 'lidar'

    trajdict = enumerate_trajs(data_root_dir, data_folders)
    if env_folders[0] == '': # env_folders not specified use all envs
        env_folders = list(trajdict.keys())

    print('*** LiDAR generation start ***')
    for env_folder in env_folders:
        env_dir = os.path.join(data_root_dir, env_folder)
        trajlist = trajdict[env_folder]
        print('Working on env {}'.format(env_dir))

        for trajdir in trajlist:
            trajfulldir = os.path.join(env_dir, trajdir)
            print('    Move into trajectory {}'.format(trajdir))
            outDir = os.path.join(trajfulldir, lidar_folder)
            if not os.path.isdir(outDir):
                os.makedirs(outDir)

            fnList = read_depth_files(trajfulldir)
            
            sample_args = zip(fnList, [outDir, ] * len(fnList))
            # sample_wrapper((fnList[0], outDir)) # for debug
            try:
                with multiprocessing.Pool(process_num) as p:
                    p.map(sample_wrapper, sample_args)
                p.join()
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers.")
                p.terminate()
                p.join()
            
        print('*** Finished env {} ***'.format(env_dir))
        print('')
