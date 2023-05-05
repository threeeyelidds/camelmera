'''
This file is to generate the IMU for the whole dataset
'''
import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

from os.path import join, isdir
from os import mkdir
import numpy as np

from scipy import interpolate
from scipy.spatial.transform import Rotation, RotationSpline
import yaml

def interpolate_translation(time, data, ips=100):
    time_interpolate = np.arange(round(time.max() * ips)) / ips
    pose = []
    vel = []
    accel = []

    for i in range(3):
        x = data[:,i]
        tck = interpolate.splrep(time, x, s = 0, k = 4)
        x_new = interpolate.splev(time_interpolate, tck, der=0)
        vel_new = interpolate.splev(time_interpolate, tck, der = 1)
        accel_new = interpolate.splev(time_interpolate, tck, der = 2)
        pose.append(x_new)
        vel.append(vel_new)
        accel.append(accel_new)
    accel = np.array(accel).T
    vel = np.array(vel).T
    pose = np.array(pose).T
    return time_interpolate, accel, vel, pose

def interpolate_rotation(time, data, ips = 100):
    rotations = Rotation.from_quat(data)
    spline = RotationSpline(time, rotations)

    time_interpolate = np.arange(round(time.max() * ips)) / ips
    angles = spline(time_interpolate).as_euler('XYZ', degrees=False)
    angular_rate = spline(time_interpolate, 1)
    angular_acceleration = spline(time_interpolate, 2)

    return time_interpolate, angular_acceleration, angular_rate, angles

def interpolate_traj(time, data, gravity = None, ips = 100):
    '''
    '''
    if gravity is None:
        gravity = np.zeros((3,))
    time_interpolate, accel, vel, pose = interpolate_translation(time,data[:,:3],ips=ips)

    time_interpolate, angular_accel, rate, angles = interpolate_rotation(time,data[:,3:],ips=ips)
    rotations = Rotation.from_euler("XYZ", angles, degrees=False)
    angle_Mat = rotations.as_matrix()

    # angle_Mat_inv = np.linalg.inv(angle_Mat)

    accel_body = np.matmul(np.expand_dims(accel+gravity,1), angle_Mat).squeeze(1)
    vel_body = np.matmul(np.expand_dims(vel,1),angle_Mat).squeeze(1)

    accel_body_nograv = np.matmul(np.expand_dims(accel,1), angle_Mat).squeeze(1)
    
    return time_interpolate, accel_body, vel, pose, rate, angles, vel_body, accel, accel_body_nograv

def generate_imudata(posefile, outputdir, img_fps = 10, imu_fps = 100):
    """
    """
    gravity = np.array([0, 0, -9.8])
    # import ipdb;ipdb.set_trace()
    # Load file
    poses = np.loadtxt(posefile,dtype = np.float32)
    length = poses.shape[0]
    img_time = np.float32(np.arange(length))/img_fps
    
    # Fit data
    imu_time, accel_body, vel, pose, rate, angles, vel_body, accel_nograv, accel_nograv_body = interpolate_traj(img_time, poses, gravity = gravity, ips = imu_fps)

    if not isdir(outputdir):
        mkdir(outputdir)

    np.savetxt(join(outputdir,"acc.txt"),accel_body)
    np.savetxt(join(outputdir,"gyro.txt"),rate)
    np.savetxt(join(outputdir,"imu_time.txt"),imu_time)
    np.savetxt(join(outputdir,"cam_time.txt"),img_time)
    np.savetxt(join(outputdir,"vel_global.txt"), vel)
    np.savetxt(join(outputdir,"vel_body.txt"), vel_body)
    np.savetxt(join(outputdir,"pos_global.txt"), pose)
    np.savetxt(join(outputdir,"ori_global.txt"), angles)
    np.savetxt(join(outputdir,"acc_nograv.txt"), accel_nograv)
    np.savetxt(join(outputdir,"acc_nograv_body.txt"), accel_nograv_body)

    np.save(join(outputdir,"acc"),accel_body)
    np.save(join(outputdir,"gyro"),rate)
    np.save(join(outputdir,"imu_time"),imu_time)
    np.save(join(outputdir,"cam_time"),img_time)
    np.save(join(outputdir,"vel_global"), vel)
    np.save(join(outputdir,"vel_body"), vel_body)
    np.save(join(outputdir,"pos_global"), pose)
    np.save(join(outputdir,"ori_global"), angles)
    np.save(join(outputdir,"acc_nograv"), accel_nograv)
    np.save(join(outputdir,"acc_nograv_body"), accel_nograv_body)

    with open(join(outputdir, 'parameter.yaml'), 'w') as f:
        params = {'img_fps': img_fps, 'imu_fps': imu_fps}
        yaml.dump(params, f)
    
# example:
# python -m postprocessing.imu_generator 
#   --data-root /home/amigo/tmp/test_root 
#   --env-folders downtown2

if __name__ == '__main__':
    from settings import get_args
    from .data_enumeration import enumerate_trajs

    args = get_args()
    data_root_dir   = args.data_root
    data_folders    = args.data_folders.split(',')
    env_folders     = args.env_folders.split(',')

    img_fps         = args.image_fps
    imu_fps         = args.imu_fps
    imu_folder      = args.imu_outdir
    imu_posefile    = args.imu_input_posefile

    trajdict = enumerate_trajs(data_root_dir, data_folders)
    if env_folders[0] == '': # env_folders not specified use all envs
        env_folders = list(trajdict.keys())

    print('*** IMU generation start ***')
    # import ipdb;ipdb.set_trace()
    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        trajlist = trajdict[env_folder]
        print('Working on env {}'.format(env_dir))

        for trajdir in trajlist:
            trajfulldir = join(env_dir, trajdir)
            print('    Move into trajectory {}'.format(trajdir))
            posefile = join(trajfulldir, imu_posefile)
            outputdir = join(trajfulldir, imu_folder)
            if not isdir(outputdir):
                mkdir(outputdir)
            generate_imudata(posefile, outputdir, img_fps = img_fps, imu_fps = imu_fps)
        
        print('*** Finished env {} ***'.format(env_dir))
        print('')
