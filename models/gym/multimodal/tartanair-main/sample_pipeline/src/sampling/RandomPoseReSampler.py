from threading import current_thread
import numpy as np
import matplotlib.pyplot as plt
import scipy

if scipy.__version__ >= '1.3.1':
    # New scipy.
    # https://www.holadevs.com/pregunta/106541/importerror-cannot-import-name-spline-from-scipyinterpolate
    from scipy.interpolate import make_interp_spline
    def spline_impl(x, y, x_smooth):
        return make_interp_spline(x, y)(x_smooth)
else:
    from scipy.interpolate import spline
    def spline_impl(x, y, x_smooth):
        return spline(x, y, x_smooth)

# from mpl_toolkits.mplot3d import *

np.set_printoptions(suppress=True, precision=2)

class RandomwalkPoseReSampler(object):
    '''
    sample poses from path with random distance
    '''

    def __init__(self, max_vel=10.0, min_vel=0.0, camera_rate=10., max_acc=30.):
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.camera_rate = camera_rate
        self.max_acc = max_acc

    def pos_distance(self, pos1, pos2):
        diff = np.array(pos1) - np.array(pos2)
        return np.linalg.norm(diff, axis=0)

    def pt_interpolate(self, pt1, pt2, loc):
        '''
        loc -> [0, 1)
        '''
        return np.array([pt1[0] * (1-loc) + pt2[0] * loc,
                pt1[1] * (1-loc) + pt2[1] * loc,
                pt1[2] * (1-loc) + pt2[2] * loc])

    def smoothCurve(self, X, Fac):
        NPoints = X.shape[0]
        dim = X.shape[1]
        idx = range(NPoints)
        idxx = np.linspace(0, NPoints, NPoints * Fac)
        Y = np.zeros((NPoints * Fac, dim))
        for ii in range(dim):
            Y[:, ii] = spline_impl(idx, X[:, ii], idxx)
        Y = Y[0:NPoints * Fac - Fac, :]
        return Y

    def sample_vel(self, vel0, ):
        while True:
            acc = self.max_acc * (2*np.random.rand() -1) # random range: -1 -> 1
            vel = vel0 + acc / self.camera_rate
            if vel < self.max_vel and vel > self.min_vel:
                return vel, acc

    def plot3d(self, positions, positions2=None, ind=None):
        # from mpl_toolkits.mplot3d import *

        positions = np.array(positions)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20, c='r')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], s=50, c='g')
        if ind is not None:
            ax.scatter(positions[ind, 0], positions[ind, 1], positions[ind, 2], s=50, c='b')
        ax.set_xlabel('X Label')
        if positions2 is not None:
            ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], s=2, c='b')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # plt.show()

    def plotvelocity(self, velocity, acceleration):
        plt.figure()
        plt.subplot(221)
        velocity = velocity
        plt.plot(velocity,'.-')
        plt.subplot(222)
        plt.hist(velocity, bins=100)
        # acceleration = np.gradient(velocity) * camera_rate
        # acceleration = acceleration[2:len(acceleration)]
        plt.subplot(223)
        plt.plot(acceleration,'.-')
        plt.subplot(224)
        plt.hist(acceleration, bins=100)

    def find_nextpos(self, sampledist, dist_list, segind, segpos):
        '''
        sampledist: distance sampled along the path
        dist_list: distance list 
        segind: current segment index
        segpos: starting point on the segment
        '''
        pathlen = len(dist_list) 
        remainSeglen = dist_list[segind]  * (1-segpos)

        while(sampledist>=remainSeglen): # sample next segment
            segind += 1
            if segind  == pathlen: # end of the path
                return pathlen, 0.0
            sampledist -= remainSeglen
            remainSeglen = dist_list[segind] 
            segpos = 0

        segpos = segpos + sampledist/dist_list[segind]
        return segind, segpos

    def resample_by_length(self, positionlist, lengthlist):
        '''
        postionlist: pos0, pos1, ..., posk
        lengthlist: len0, len1, ..., lenj
        assume: sum(len) == sum(dist(pos_i, pos_{i+1}))
        return: new_positionlist, where newpos0==pos0, newpos{j+1}==posk, dist(newpos_i, newpos_{i+1}==len_i)
        '''
        distlist = [self.pos_distance(positionlist[i], positionlist[i - 1]) 
                            for i in range(1, len(positionlist))]
        assert abs(sum(distlist) - sum(lengthlist)) < 1e-6
        newposlist = [positionlist[0]]
        segind, segpos = 0, 0.0
        for seglen in lengthlist[:-1]:
            segind, segpos = self.find_nextpos(seglen, distlist, segind, segpos)
            pos_new = self.pt_interpolate(positionlist[segind], positionlist[segind+1], segpos) 
            newposlist.append(pos_new)
        newposlist.append(positionlist[-1])
        return newposlist

    def acc_verify(self, positionlist):
        positionlist = np.array(positionlist)
        vellist = (positionlist[1:] - positionlist[:-1]) * self.camera_rate
        acclist = (vellist[1:] - vellist[:-1]) * self.camera_rate
        # print(acclist, outliers+2)
        accok = acclist.max() < self.max_acc and acclist.min() > -self.max_acc
        # assert len(acclist)==1
        return accok, acclist # 

    def pose_resample(self, position_list, dist_list, visualize=False, smooth_modify_thresh=0.05):
        '''
        given a sequence of positions, generate random walk velocity and resample the position on the trajectory
        return: N x 7 array [location[0] (the starting index), location[1] (the percentage), pose_x, pose_y, pose_z, velocity, acceleration]
        x---------x--------------x-------x---
        loc[0]   loc[1]         loc[2]  loc[3]
        pos[0]   pos[1]         pos[2]  pos[3]
        0.0  vel[1]      vel[2]      vel[3]
        0.0  0.0   acc[2]       acc[3]  
        0.0  velxyz[1]   velxyz[2]   velxyz[3]
        '''
        pathlen = len(position_list)

        segind, segpos = 0 ,0.0 # start_postion_ind, percentage -> [0,1)
        res = [position_list[0]]
        # import ipdb;ipdb.set_trace()
        vel = (np.random.rand() * (self.max_vel - self.min_vel) + self.min_vel) # use a random initial velocity -- vel[1]
        dist = vel / self.camera_rate
        segind, segpos = self.find_nextpos(dist, dist_list, segind, segpos)
        pos_new = self.pt_interpolate(position_list[segind], position_list[segind+1], segpos)
        res.append(pos_new) # insert the second frame

        acc_modify_count = 0
        discard_count = 0
        discardlist = []
        # outlierlist = []

        while True: 
            vel_candidate, acc_candidate = self.sample_vel(vel)
            dist_candidate = vel_candidate / self.camera_rate
            segind_candidate, segpos_candidate = self.find_nextpos(dist_candidate, dist_list, segind, segpos)
            if segind_candidate == pathlen -1: # this is the last point in the trajectory
                res.append(position_list[-1])
                pl = res[-3:]
                accok, _ = self.acc_verify(pl)
                if not accok:
                    newposelist, modifynum = self.repair_peak(res, len(res)-1)
                    if len(newposelist)==0:
                        return False, []
                    acc_modify_count += modifynum
                    res = newposelist
                break
            pos_candidate = self.pt_interpolate(position_list[segind_candidate], position_list[segind_candidate+1], segpos_candidate)
            # print(segind_next, segpos_next)
            # verify the actual acc in x,y,z direction is within the range 
            pl = [res[-2], res[-1], pos_candidate]
            accok, accxyz = self.acc_verify(pl)
            # import ipdb;ipdb.set_trace()
            if accok:  # the acceleration is good
                segind, segpos, vel = segind_candidate, segpos_candidate, vel_candidate
                res.append(pos_candidate)

                discard_count = 0
                discardlist = []
                # print(segind, segpos, vel, acc, acc_xyz)
            elif  discard_count>20: # 20 is hard coded 
                acclist = [ll[0] for ll in discardlist]
                minind = np.argmin(acclist) # select the one with the smallest acc, so that later we have a better time reparing the peak
                accxyzmax, segind_candidate, segpos_candidate, pos_candidate, vel_candidate = discardlist[minind]
                res.append(pos_candidate)
                # print("  Use smallest acc ind {} segind {}, segpos {}, acc_xyz {}".format(minind, segind_candidate, segpos_candidate, accxyzmax))

                newposelist, modifynum = self.repair_peak(res, len(res)-1)
                if len(newposelist)==0:
                    return False, []
                if visualize:
                    # import ipdb;ipdb.set_trace()
                    rrr = np.array(res)
                    poses = np.array([ll[3] for ll in discardlist])
                    plt.scatter(rrr[-modifynum-3:,0], rrr[-modifynum-3:,1], color="b", s=150)
                    plt.scatter(poses[:, 0], poses[:, 1], color="r")
                    plt.scatter(pos_candidate[0], pos_candidate[1], color="c", s=100)
                    plt.scatter(np.array(newposelist)[-modifynum-8:,0], np.array(newposelist)[-modifynum-8:,1], color="g")
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.show()
                # import ipdb;ipdb.set_trace()
                segind, segpos = segind_candidate, segpos_candidate
                vel = self.pos_distance(newposelist[-1], newposelist[-2]) * self.camera_rate
                res = newposelist
                discard_count = 0
                discardlist = []
                acc_modify_count += modifynum

            else: # sample a few times to see if we can get lucky
                discard_count += 1
                discardlist.append([np.abs(accxyz).max(), segind_candidate, segpos_candidate, pos_candidate, vel_candidate])
                # print("Discard {} point with large acc {}, segind {}, segpos {}".format(discard_count, accxyz, segind_candidate, segpos_candidate))

        print("Sample complete, modified {}/{}".format(acc_modify_count, len(res)))
        suc = float(acc_modify_count)/len(res) < smooth_modify_thresh
        return suc, res 

    def repair_peak(self, poselist, peakind):
        '''
        poselist: N x 3 numpy array
        vellist: N x 3 numpy array
        insert position to the left/right of the peakind, using arithmetic progression
        given k (samplenum): the sequence length
              s (seg_total_len): the sum of the seqence
              s_k (sublenlist[-1]): the speed of the end point (connection point)
        calculate d: the common difference between terms
                  s1: the speed of the peak
        d = (s_k*k -s)*2/(k-1)/k                  
        s1 = 2*s/k - s_k
        constraints: s_k>s/k
                     s_k<2*s/k
        x------------------------x--------------x-----------------------x
        peak-3                  peak-2         peak-1                  peak
        |                        |                     |                |
        new(peak-3)             new(peak-2)           new(peak-1)      new(peak)
        '''
        # fix the left side of the peak
        listlen = len(poselist)
        posnum = 3
        samplenum = 3
        while True:
            if peakind - posnum < 0: # not enough pos on the left side
                return [], None
            subposlist = [poselist[peakind-k] for k in range(posnum+1)] # len: posnum+1, [pos[peak], pos[peak-1],...,pos[peak-k]]
            sublenlist = [self.pos_distance(subposlist[i], subposlist[i - 1]) 
                                for i in range(1, len(subposlist))] # len: posenum
            seg_total_len = sum(sublenlist)
            # print(sublenlist)
            if sublenlist[-1] * samplenum < seg_total_len:
                # print('increase samplenum')
                samplenum += 1
                # TODO: add a length check in case solution cannot be found?
                continue
            if sublenlist[-1] * samplenum > 2*seg_total_len:
                # print('go to next point')
                posnum += 1
                samplenum = posnum
                continue
            new_lenlist, d_len = arithmetic_progression(sublenlist[-1], seg_total_len, samplenum)
            newposlist = self.resample_by_length(subposlist, new_lenlist)
            assert(self.pos_distance(subposlist[-2],newposlist[-2])<1e-6) # peak-2 should be at the same place
            # print(new_lenlist)
            # print(subposlist)
            # print(newposlist)
            # self.acc_verify(subposlist[::-1])
            accok, _ = self.acc_verify(newposlist[::-1])
            if not accok:
                samplenum += 1
            else:
                break

        # print("  Smooth left side {} frames with {} new frames".format(posnum, samplenum))
        resposlist = poselist[:peakind-posnum] + newposlist[::-1]
        # if peakind + 1 >= listlen-1: # already the last frame
        #     resposlist = np.concatenate((resposlist, poselist[peakind+1:]), axis=0)
        # import ipdb;ipdb.set_trace()
        return resposlist, samplenum

    def sample_poses(self, path, visfilename='', smooth_modify_thresh=0.05):
        '''
        path is a list of positions [[x0,y0,z0], [x1,y1,z1],...]
        re-sample the poses along the path with random velocity
        '''

        # interpolate position to a smooth curve
        path = np.array(path)
        print('number of initial positions is = {0:}'.format(len(path)))

        smooth_positions = self.smoothCurve(path, 10)

        # calculate global and each step path length
        each_step_length = [self.pos_distance(smooth_positions[i], smooth_positions[i - 1]) 
                            for i in range(1, len(smooth_positions))]
        path_length = sum(each_step_length)

        print('path_length = {0:.2f}, mean velocity = {1:}, camera_rate = {2:})'.format(path_length,
                                                                                     0.5 * (self.max_vel + self.min_vel),
                                                                                     self.camera_rate))

        suc = False
        current_thresh = smooth_modify_thresh
        while current_thresh <= 0.2:
            for k in range(50):
                suc, res = self.pose_resample(smooth_positions, each_step_length, visualize=False, smooth_modify_thresh=current_thresh)
                if suc:
                    break
            if suc:
                break
            current_thresh += 0.05
        if not suc:
            return None

        poselist = np.array(res)

        poselist_ori = poselist.copy()
        accok, _ = self.acc_verify(poselist)
        assert accok

        # self.plot3d(positions=poselist, ind=outlierlist)
        # plt.show()

        if visfilename != '':
            # visualize the outliers in 3d to see whether they make sense
            # self.plot3d(positions=poselist_ori)
            # plt.show()

            fig = plt.figure(figsize=(15,12))
            ax = fig.add_gridspec(2, 3)
            poselist = np.array(poselist)
            velxyz = poselist[1:] - poselist[:-1]
            accxyz = velxyz[1:] - velxyz[:-1]
            ax1 = fig.add_subplot(ax[0, 0:2])
            ax1.plot(velxyz, '.')
            ax1.set_title('Velocity xyz')

            ax2 = fig.add_subplot(ax[1, 0:2])
            ax2.plot(accxyz, '.')
            ax2.set_title('Acceleration xyz')
            # plt.show()

            # visualize the peak smooth
            velocity = [self.pos_distance(poselist[k], poselist[k+1]) for k in range(len(poselist)-1) ]
            acceleration = np.array(velocity)[1:] - np.array(velocity)[:-1]
            ax3 = fig.add_subplot(ax[0, 2])
            ax3.hist(velocity, bins=100)
            ax4 = fig.add_subplot(ax[1, 2])
            ax4.hist(acceleration, bins=100)

            plt.savefig(visfilename)

        return res

def arithmetic_progression(at, s, k):
    '''
    at: the last number in the sequence
    s: the sum of the sequence
    k: the length of the sequence
    return: the sequence [s1,s2,..,s{k-1}] 
            d: difference
    '''
    d = (at * k - s) * 2 / k / (k-1)
    slist = [at - kk * d for kk in range(k)]

    return slist[::-1], d

if __name__ == '__main__':
    MaxVelo = 10.0
    MinVelo = 0.0
    CameraRate = 10.0
    Max_acc = 20.0
    trajfile = '/home/amigo/workspace/ros_tartanair/src/resample/ros_path_0723_210147/P002.txt'
    traj_np = np.loadtxt(trajfile)

    posesampler = RandomwalkPoseReSampler(max_vel=MaxVelo, min_vel=MinVelo, camera_rate=CameraRate, max_acc=Max_acc)
    positions = posesampler.sample_poses(traj_np.tolist(), visualize=True)
    print(positions)

