# collect images in cv mode
# import os
# import sys
# # The path of the current Python script.
# _CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# sys.path.append(os.path.join(_CURRENT_PATH, '..'))

import numpy as np
from math import pi
from scipy.spatial.transform import Rotation, RotationSpline

np.set_printoptions(precision=3, suppress=True, threshold=10000)


class QuaternionSampler(object):
    '''
    Use scipy Rotation for spline smoothing
    angles are in the order of yaw, pitch, roll
    '''
    def __init__(self, rand_degree, smooth_count, max_yaw=180., min_yaw=-180., max_pitch=90., min_pitch=-90., max_roll=90., min_roll=-90.):
        # self.MaxRandStepAngle = self.args.rand_degree # np.clip(self.args.rand_degree, 0, 90)
        self.MaxRandRad = rand_degree * pi / 180.
        self.SmoothCount =  smooth_count
        # self.MaxRandSampleRad = self.MaxRandRad * self.SmoothCount
        # assert self.MaxRandSampleRad < pi/2, 'The sample param: rand-degree {}, and smooth-count {} cannot be achieved'.format(self.MaxRandStepAngle, self.SmoothCount)

        self.MaxYaw = min(max_yaw, 180) 
        self.MinYaw = max(min_yaw, -180) 
        self.MaxPitch = min(max_pitch, 90) 
        self.MinPitch = max(min_pitch, -90) 
        self.MaxRoll = min(max_roll, 180) 
        self.MinRoll = max(min_roll, -180) 

    def sample_orientations(self, framenum, visfilename=''):
        pose_angle = self.init_random_yaw()
        pose_rot = Rotation.from_euler("ZYX", pose_angle, degrees=True)
        anglelist = [pose_angle]
        quatlist = [pose_rot.as_quat()]
        # thetalist = []
        # axislist = []
        # rot_list = []
        
        for k in range(framenum-1):
            good_angle = False
            while not good_angle:
                rand_rot, theta, axis = self.random_rotation()
                pose_rot_candidate = rand_rot * pose_rot
                pose_angle_candidate = pose_rot_candidate.as_euler("ZYX", degrees=True)
                good_angle = self.good_angle(pose_angle_candidate)
            pose_angle = pose_angle_candidate
            pose_rot = pose_rot_candidate
            anglelist.append(pose_angle)
            quatlist.append(pose_rot.as_quat())
            # thetalist.append(theta)
            # axislist.append(axis)
            # rot_list.append(pose_rot)
            # pose_angle_clip = self.clip_angle(pose_angle)
            # pose_rot = Rotation.from_euler("ZYX", pose_angle_clip, degrees=True)
            # pose_angle_debug = pose_rot.as_euler("ZYX", degrees=True)
            # print(pose_angle, pose_angle_clip, pose_angle_debug)
            # anglelist.append(pose_angle_clip)

        angles = np.array(anglelist)
        quats = np.array(quatlist)
        quats_smooth, angles_smooth = self.traj_smoothing(quats)
        self.traj_analyze(quats_smooth)

        if visfilename != '':
            self.traj_vis(angles, quats, angles_smooth, quats_smooth, visfilename)

        return angles_smooth, quats_smooth #thetalist, axislist, rot_list
        
    def traj_smoothing(self, quatlist):
        # import ipdb;ipdb.set_trace()
        # skip sample the steps according to the smooth-count
        seqlen = len(quatlist)
        # timestamp for key frames
        times = list(range(0, seqlen, self.SmoothCount))
        if times[-1] != seqlen-1: # add the last frame to make sure it is interpolate
            times.append(seqlen-1)
        sample_quatlist = quatlist[times]

        # smooth the trajectory
        rotations_quats = Rotation.from_quat(np.array(sample_quatlist))
        spline_quats = RotationSpline(times, rotations_quats)
        quatlist_smooth = spline_quats(np.array(range(seqlen))).as_quat()
        anglist_smooth = spline_quats(np.array(range(seqlen))).as_euler('ZYX', degrees=True)

        return quatlist_smooth, anglist_smooth

    def init_random_yaw(self, yaw_first = True):
        randomyaw = np.random.uniform(self.MinYaw, self.MaxYaw)
        print ('Init orientation: Random yaw {}, angle {}'.format(randomyaw*pi/180., randomyaw))
        if yaw_first:
            return [randomyaw, 0.0, 0.0]    
        else:
            return [0.0, 0.0, randomyaw]  # 

    def random_rotation(self):
        '''
        return Quaternionpy for multiplication
        '''
        theta =  (np.random.random()*2 - 1) * self.MaxRandRad
        axis = np.random.random(3)
        axis = axis/np.linalg.norm(axis)

        quat=np.zeros(4)
        quat[0:3] = np.sin(theta/2)*axis
        quat[3] = np.cos(theta/2)

        return Rotation.from_quat(quat), theta, axis # for debug

    def good_angle(self, angle):
        '''
        Input quatpy: Quaternionpy
        Return new_ori_clip: Quaternionpy
        '''
        (yaw, pitch, roll) = angle
        # (roll, pitch, yaw  ) = angle
        if pitch < self.MinPitch or pitch > self.MaxPitch:
            return False
        if roll < self.MinRoll or roll > self.MaxRoll:
            return False
        if yaw < self.MinYaw or yaw > self.MaxYaw:
            return False

        return True

    def traj_analyze(self, quats):
        framenum = len(quats)
        thetalist = []
        rots = Rotation.from_quat(quats)
        mats = rots.as_matrix()
        for k in range(1,framenum):
            # quat1 = quats[k-1]
            # quat2 = quats[k]
            # rot1 = Rotation.from_quat(quat1)
            # rot2 = Rotation.from_quat(quat2)
            # mat1 = rot1.as_matrix()
            # mat2 = rot2.as_matrix()
            mat1 = mats[k-1]
            mat2 = mats[k]
            delta_mat = mat1.transpose() @ mat2
            delta_vec = Rotation.from_matrix(delta_mat).as_rotvec()
            delta_theta = np.linalg.norm(delta_vec)
            thetalist.append(delta_theta)
        thetalist = np.array(thetalist)
        print('Seqlen {}, Smooth {}, Max Deg {:.2f}, Max/Mean theta {:.2f}/{:.2f}'.format(len(quats), self.SmoothCount, self.MaxRandRad*180./pi, thetalist.max()*180./pi, thetalist.mean()*180./pi))

    def traj_vis(self, angles, quats, angles_smooth, quats_smooth, visfilename):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15,12))
        ax = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(ax[0, 0:3])
        ax1.plot(angles, 'x')
        ax1.plot(angles_smooth, '-')
        ax1.grid()
        ax1.set_title('Angle')
        ax1.legend(['yaw','pitch','roll'])

        ax2 = fig.add_subplot(ax[1, 0:3])
        ax2.plot(quats, 'x')
        ax2.plot(quats_smooth, '-')
        ax2.grid()
        ax2.set_title('Quaternion')
        ax2.legend(['x','y','z','w'])

        ax3 = fig.add_subplot(ax[2, 0])
        rate = angles_smooth[1:, 0] - angles_smooth[:-1, 0]
        rate[rate<-180] = rate[rate<-180] + 360
        rate[rate>180] = rate[rate>180] - 360
        ax3.hist(rate, bins = 50)
        ax3.set_title('Angular rate x')

        ax4 = fig.add_subplot(ax[2, 1])
        rate = angles_smooth[1:, 1] - angles_smooth[:-1, 1]
        rate[rate<-180] = rate[rate<-180] + 360
        rate[rate>180] = rate[rate>180] - 360
        ax4.hist(rate, bins = 50)
        ax4.set_title('Angular rate y')

        ax5 = fig.add_subplot(ax[2, 2])
        rate = angles_smooth[1:, 2] - angles_smooth[:-1, 2]
        rate[rate<-180] = rate[rate<-180] + 360
        rate[rate>180] = rate[rate>180] - 360
        ax5.hist(rate, bins = 50)
        ax5.set_title('Angular rate z')

        plt.savefig(visfilename)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    class ARGS:
        def __init__(self) -> None:
            self.rand_degree  = 50
            self.smooth_count = 5
            self.max_yaw =  180
            self.min_yaw  = -180
            self.max_pitch  = 90
            self.min_pitch  = -90
            self.max_roll  = 45
            self.min_roll  = -45


    # for k in range(1000):
    #     yaw, pitch, roll = (random.random()*2-1)*180,(random.random()*2-1)*90,(random.random()*2-1)*90
    #     rot = Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True)
    #     yaw2, pitch2, roll2  = rot.as_euler("ZYX", degrees=True)
    #     print((yaw, pitch, roll), (yaw2, pitch2, roll2))
    #     if abs(yaw-yaw2)>10e-4 or abs(pitch-pitch2)>10e-4 or abs(roll-roll2)>10e-4:
    #         import ipdb;ipdb.set_trace()

    framenum = 50

    args = ARGS()
    randomsampler = QuaternionSampler(args.rand_degree, args.smooth_count, args.max_yaw, args.min_yaw, args.max_pitch, args.min_pitch, args.max_roll, args.min_roll)
    anglelist, quatlist = randomsampler.sample_orientations(framenum)

    angles = np.array(anglelist)
    quats = np.array(quatlist)
    quats_smooth, angles_smooth = randomsampler.traj_smoothing(quats)
    randomsampler.traj_analyze(quats_smooth)

    plt.plot(angles, 'x')
    plt.plot(angles_smooth, '.-')
    plt.grid()
    plt.legend(['yaw','pitch','roll'])

    plt.figure(figsize=(10,8))
    plt.plot(quats, 'x')
    plt.plot(quats_smooth, '.-')
    plt.grid()
    plt.legend(['x','y','z','w'])
    plt.show()


    # test the spline interpolation
    seqlen = len(quats_smooth)
    # timestamp for key frames
    times = list(range(seqlen))

    # smooth the trajectory
    times2 = np.arange(0,seqlen,0.2)
    rotations_quats = Rotation.from_quat(np.array(quats_smooth))
    spline_quats = RotationSpline(times, rotations_quats)
    quatlist_smooth = spline_quats(times2).as_quat()
    anglist_smooth = spline_quats(times2).as_euler('ZYX', degrees=True)

    plt.plot(angles, 'x')
    plt.plot(angles_smooth, '-')
    plt.plot(times2, anglist_smooth, '.')
    plt.grid()
    plt.legend(['yaw','pitch','roll'])

    plt.figure(figsize=(10,8))
    plt.plot(quats, 'x')
    plt.plot(quats_smooth, '-')
    plt.plot(times2, quatlist_smooth, '.')
    plt.grid()
    plt.legend(['x','y','z','w'])
    plt.show()
    

    # rate = angles[1:]-angles[:-1]
    # rate[rate<-180] = rate[rate<-180] + 360
    # rate[rate>180] = rate[rate>180] - 360

    # for k in range(1,framenum):
    #     angle1 = anglelist[k-1]
    #     angle2 = anglelist[k]
    #     rot1 = Rotation.from_euler("ZYX", angle1, degrees=True)
    #     rot2 = Rotation.from_euler("ZYX", angle2, degrees=True)
    #     mat1 = rot1.as_matrix()
    #     mat2 = rot2.as_matrix()

    #     delta_mat = mat1.transpose() @ mat2
    #     delta_vec = Rotation.from_matrix(delta_mat).as_rotvec()
    #     delta_theta = np.linalg.norm(delta_vec)

    #     print(k, delta_theta, thetalist[k-1], angles[k-1], angles[k], rate[k-1])
    #     # import ipdb;ipdb.set_trace()

    # plt.plot(angles, '.-')
    # plt.grid()
    # plt.legend(['yaw','pitch','roll'])

    # # import ipdb;ipdb.set_trace()
    # fig = plt.figure(figsize=(10,8))
    # # print(angles)
    # # print(rate)
    # plt.plot(rate, '.')
    # plt.legend(['yaw','pitch','roll'])

    # plt.show()
    import ipdb;ipdb.set_trace()

