#!/usr/bin/env python
# license removed for brevity

import sys
import os
curdir = os.path.dirname(os.path.realpath(__file__))
print(curdir)
sys.path.insert(0,curdir+'/..')
sys.path.insert(0,curdir)
# import sys
# sys.path.append('../')

# import copy
import rospy
import tf
import time
import math
import numpy as np
import expo_utility as expo_util

import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim import utils as sim_util
from airsim.utils import to_quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from planner_base.srv import path as PathSrv
from expo_base.srv import nearfrontier as NearfrontierSrv
from expo_base.srv import globalfrontier as GlobalfrontierSrv
from expo_base.srv import frontierlock as LockfrontierSrv
from expo_base.srv import pointstate as PointStateSrv

from geometry_msgs.msg import Point

from settings import get_args
from os.path import isdir, join
from os import mkdir, system


class ExpoController:
    def __init__(self, args):
        rospy.init_node('expo_control', anonymous=True)

        self.args = args
        self.cmd_client = airsim.VehicleClient(args.airsim_ip)
        self.cmd_client.confirmConnection()
        self.tf_broad_ = tf.TransformBroadcaster()
        self.odom_pub_ = rospy.Publisher('pose', Odometry, queue_size=1)
        self.cloud_pub_ = rospy.Publisher('cloud_in', PointCloud2, queue_size=1)
        self.camid = "0"
        self.img_type = [airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)]
        self.FAR_POINT = args.far_point
        self.cam_pos = [0., 0., 0.]
        self.fov = args.camera_fov
        self.path_skip = args.path_skip
        self.last_list_len = 10
        self.last_ten_goals = [[0.,0.,0.]]*self.last_list_len # detect and avoid occilation
        self.lfg_ind = 0
        self.replan_step = 1
        self.mapfilename = self.args.map_filename
        self.mapfiledir = self.args.map_dir
        self.try_round = self.args.try_round
        self.last_tgt = [-1., -1., -1.]

        if not isdir(self.mapfiledir):
            mkdir(self.mapfiledir)


    def get_depth_campos(self):
        '''
        cam_pose: 0: [x_val, y_val, z_val] 1: [x_val, y_val, z_val, w_val]
        '''
        img_res = self.cmd_client.simGetImages(self.img_type)
        
        if img_res is None or img_res[0].width==0: # Sometime the request returns no image
            return None, None

        # import ipdb;ipdb.set_trace()
        depth_front = sim_util.list_to_2d_float_array(img_res[0].image_data_float,
                                                      img_res[0].width, img_res[0].height)
        depth_front[depth_front>self.FAR_POINT] = self.FAR_POINT
        cam_pose = (img_res[0].camera_position, img_res[0].camera_orientation)

        return depth_front, cam_pose

    def collect_points_6dir(self, tgt):
        # must be in CV mode, otherwise the point clouds won't align
        scan_config = [0, -1, 2, 1, 4, 5] # front, left, back, right, up, down
        points_6dir = np.zeros((0, self.imgwidth, 3), dtype=np.float32)
        for k,face in enumerate(scan_config): 
            if face == 4:  # look upwards at tgt
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(math.pi / 2, 0, 0)) # up
            elif face == 5:  # look downwards
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(-math.pi / 2, 0, 0)) # down - pitch, roll, yaw
            else:  # rotate from [-90, 0, 90, 180]
                yaw = math.pi / 2 * face
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, yaw))

            # import ipdb;ipdb.set_trace()
            self.set_vehicle_pose(pose)
            depth_front, _ = self.get_depth_campos()
            if depth_front is None:
                rospy.logwarn('Missing image {}: {}'.format(k, tgt))
                continue
            # import ipdb;ipdb.set_trace()
            point_array = expo_util.depth_to_point_cloud(depth_front, self.focal, self.pu, self.pv, mode = k)
            points_6dir = np.concatenate((points_6dir, point_array), axis=0)
        # reset the pose for fun
        pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, 0))
        self.set_vehicle_pose(pose)
        # print 'points:', points_6dir.shape            
        return points_6dir

    def publish_lidar_scans_6dir(self, tgt):
        # if the same target, don't publish the points again
        if self.is_same_point(tgt, self.last_tgt):
            rospy.loginfo('Skip pc publishing for the same position!')
            return
        # import ipdb;ipdb.set_trace()
        rostime = rospy.Time.now()
        points = self.collect_points_6dir(tgt)
        pc_msg = expo_util.xyz_array_to_point_cloud_msg(points, rostime)
        odom_msg = expo_util.trans_to_ros_odometry(tgt, rostime)
        self.cam_pose = tgt

        self.tf_broad_.sendTransform(translation=tgt, rotation=[0.,0.,0.,1.],
                                     time=rostime, child='map', parent='world')
        self.odom_pub_.publish(odom_msg)
        self.cloud_pub_.publish(pc_msg)

        self.last_tgt = tgt

    def set_vehicle_pose(self, pose, ignore_collison=True, vehicle_name=''):
        self.cmd_client.simSetVehiclePose(pose, ignore_collison, vehicle_name) # amigo: this is supposed to be used in CV mode
        time.sleep(0.1)

    def init_exploration(self):
        # cloud_msg, cloud_odom_msg = self.get_point_cloud_msg(cam_id=0)
        # cam_trans, cam_rot = expo_util.odom_to_trans_and_rot(cloud_odom_msg)
        # cam_pose = self.cmd_client.simGetCameraInfo(camera_name=0).pose
        cam_info = self.cmd_client.simGetCameraInfo(camera_name=self.camid)
        img_res = self.cmd_client.simGetImages(self.img_type)
        img_res = img_res[0]
        cam_pose = Pose(img_res.camera_position, img_res.camera_orientation)
        cam_trans, cam_rot = expo_util.sim_pose_to_trans_and_rot(cam_pose)
        self.imgwidth = img_res.width
        self.imgheight = img_res.height
        self.focal, self.pu, self.pv = expo_util.get_intrinsic(img_res, cam_info, self.fov)
        self.cam_pos = cam_trans

        rospy.loginfo('Initialized img ({},{}) focal {}, ({},{})'.format(self.imgwidth, self.imgheight, self.focal, self.pu, self.pv))
        self.publish_lidar_scans_6dir(cam_trans)
        time.sleep(5)

    def points_dist(self, pt1, pt2):
        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
        dist = math.sqrt(dist)
        return dist

    def is_same_point(self, pt1, pt2):
        if abs(pt1[0]-pt2[0]) > 1e-3 or abs(pt1[1]-pt2[1]) > 1e-3 or abs(pt1[2]-pt2[2]) > 1e-3:
            return False
        return True

    def explore_replan(self):
        """
        depracated
        We have to handle two cases: oscillation and local frontier disappear
        :param try_round: -1 for all the frontiers
        :path_skip: skip way points on the path
        :return:
        """
        # import ipdb;ipdb.set_trace()
        # A star path finding for local exploration
        local_path = self.call_local_planning_service(self.try_round)
        if local_path is None: 
            return False
        # import ipdb;ipdb.set_trace()
        # No feasible local_path is found
        if len(local_path) == 0: 
            return False

        # insert the goal point to the list for occilation detection
        target_pt = local_path[0]
        occilation = False
        if not self.is_same_point(target_pt, self.last_ten_goals[self.lfg_ind]): # target point changes
            for k,pt in enumerate(self.last_ten_goals): # check if the target already exist in the goal list
                if self.is_same_point(target_pt, pt): # this is a occilation
                    occilation = True
                    tmp = self.last_ten_goals[self.lfg_ind]
                    self.last_ten_goals[self.lfg_ind] = pt
                    self.last_ten_goals[k] = tmp
                    rospy.logwarn('Occilation detected, increase replan step! %d - %d', self.lfg_ind, k)
                    break
            if occilation:
                self.replan_step = self.replan_step * 3
                if self.replan_step>7: # serious occilation
                    rospy.logwarn('Escapte occilation by going the the goal directly!!')
                    local_path = [local_path[0]]
            else:
                rospy.loginfo("new target, reset replan step..")
                self.lfg_ind = (self.lfg_ind+1)%self.last_list_len
                self.last_ten_goals[self.lfg_ind] = target_pt
                self.replan_step = 1
        else:
            rospy.loginfo("flying to the same target")

        path_len = len(local_path)
        for i in range(self.replan_step):
            if path_len < self.path_skip:
                next_ind = path_len - (i+1)*((path_len+1)/2) 
            else:
                next_ind = len(local_path) - (i+1)*self.path_skip 
            if next_ind<0:
                break
            rospy.loginfo('Path len {}, move to waypoint {}'.format(len(local_path),local_path[next_ind]))
            next_way_point = local_path[next_ind]
            # self.move_to_tgt(next_way_point)
            self.publish_lidar_scans_6dir(next_way_point)
            time.sleep(2.0)

        return True

    def move_along_path(self, plan_path):
        # import ipdb;ipdb.set_trace()
        path_len = len(plan_path)
        next_ind = max(1, path_len - self.path_skip)
        if path_len==1:
            rospy.logwarn('Pathlen is only 1!') # TODO: Do something in ths situation? Delete the frontier? 
            return 
        while next_ind > 0: # skip a few steps and move to the next point on the path
            # import ipdb;ipdb.set_trace()
            next_way_point = plan_path[next_ind]
            rospy.loginfo('Pathlen {}, Move to waypoint ind {} {}'.format(path_len, next_ind, next_way_point))
            pt_state = self.call_check_way_point_safe(next_way_point)
            if pt_state is None or pt_state == False: # TODO: delete the corresponding frontier??
                rospy.loginfo('Waypoint is occupied {}'.format(next_way_point))
                break
            self.publish_lidar_scans_6dir(next_way_point)
            time.sleep(2.0)
            self.call_wait_frontier_update_service()
            if not self.call_global_frontier_still_exists(plan_path[0]):
                rospy.loginfo('The frontier has been cleared! ')
                break
            if next_ind ==1: # already next to the frontier but it hasn't been cleared
                rospy.logwarn('Cannot clear the frontier!')
                break # TODO: Do something in ths situation? Delete the frontier? 
            next_ind = max(1, next_ind - self.path_skip)


    def explore(self):
        """
        :path_skip: skip way points on the path
        :return:
        """
        # A star path finding for local exploration
        local_path = self.call_local_planning_service(self.try_round)
        if local_path is None: 
            return False
        # import ipdb;ipdb.set_trace()
        # No feasible local_path is found
        if len(local_path) == 0: 
            return False

        self.move_along_path(local_path)

        return True

    def explore_global_frontier(self,):
        # import ipdb;ipdb.set_trace()
        next_point, _ = self.get_nearest_frontier()
        if next_point is None:
            return False
        rospy.loginfo('Next global frontier ({}, {}, {})'.format(next_point[0], next_point[1], next_point[2]))
        self.publish_lidar_scans_6dir(next_point)
        return True

    def teletrans_global_safepose(self,):
        next_frontier_point, safe_pose = self.get_nearest_frontier()
        if safe_pose is None: # No more global frontiers
            return False
        rospy.loginfo('Transport to safe pose ({}, {}, {}) with global frontier ({}, {}, {})'.format(
                                            safe_pose[0], safe_pose[1], safe_pose[2],
                                            next_frontier_point[0], next_frontier_point[1], next_frontier_point[2]))
        self.call_lock_global_frontier_service()
        self.publish_lidar_scans_6dir(safe_pose)
        time.sleep(2.0) # TODO: add lock to make sure planning happens after map update

        # find path from the safe pose to the global frontier
        # 1. A* a path from the pose to frontier
        #    - if path is not found, local_path is None, and the global frontier is deleted
        #    - if path is found, it is returned in local_path
        local_path = self.call_safepose_to_frontier_service(next_frontier_point)
        if local_path is not None and len(local_path) > 0:
            self.move_along_path(local_path)
        else:
            rospy.loginfo('Plan to the global frontier not found! ')

        return True

    def get_nearest_frontier(self): 
        rospy.wait_for_service('near_frontier_srv')

        robot_pos = Point(self.cam_pose[0], self.cam_pose[1], self.cam_pose[2])
        try:
            global_frontier_srv = rospy.ServiceProxy('near_frontier_srv', NearfrontierSrv)
            global_frontier_res = global_frontier_srv(robot_pos)
        except rospy.ServiceException:
            print("No frontier returned..")
            return None, None

        return [global_frontier_res.nearfrontier.x, global_frontier_res.nearfrontier.y, global_frontier_res.nearfrontier.z], \
               [global_frontier_res.safepose.x, global_frontier_res.safepose.y, global_frontier_res.safepose.z]

    def call_local_planning_service(self, try_round):
        target_path = []
        rospy.wait_for_service('bbx_path_srv')
        try:
            feasible_path = rospy.ServiceProxy('bbx_path_srv', PathSrv)
            resp = feasible_path(try_round)
            for item in resp.path.points:
                target_path.append([item.x, item.y, item.z])
            return target_path
        except rospy.ServiceException:
            rospy.loginfo("No local frontier exists, or bbx_path_srv service call failed..")
            return None

    def call_lock_global_frontier_service(self):
        rospy.wait_for_service('lock_frontier_srv')

        try:
            lock_frontier_srv = rospy.ServiceProxy('lock_frontier_srv', LockfrontierSrv)
            lock_frontier_srv()
        except rospy.ServiceException:
            rospy.logwarn("Error in lock_frontier_srv ..")

    def call_wait_frontier_update_service(self): 
        rospy.wait_for_service('wait_frontier_srv')

        try:
            wait_frontier_srv = rospy.ServiceProxy('wait_frontier_srv', LockfrontierSrv)
            wait_frontier_srv()
        except rospy.ServiceException:
            rospy.logwarn("Error in wait_frontier_srv ..")


    def call_safepose_to_frontier_service(self, frontier_point):
        target_path = []
        rospy.wait_for_service('plan_to_frontier_srv')
        try:
            feasible_path = rospy.ServiceProxy('plan_to_frontier_srv', GlobalfrontierSrv)
            frontier_pt = Point(frontier_point[0], frontier_point[1], frontier_point[2])
            resp = feasible_path(frontier_pt)
            for item in resp.path.points:
                target_path.append([item.x, item.y, item.z])
            return target_path
        except rospy.ServiceException:
            rospy.logwarn("plan_to_frontier_srv service call failed")
            return None

    def call_check_way_point_safe(self, way_point):
        rospy.wait_for_service('check_point_status')
        try:
            point_state_srv = rospy.ServiceProxy('check_point_status', PointStateSrv)
            way_point = Point(way_point[0], way_point[1], way_point[2])
            resp = point_state_srv(way_point)
            if resp.state.data:
                return True
            return False
        except rospy.ServiceException:
            rospy.logwarn("plan_to_frontier_srv service call failed")
            return None
       
    def call_global_frontier_still_exists(self, frontier_point):
        rospy.wait_for_service('is_global_frontier')
        try:
            point_state_srv = rospy.ServiceProxy('is_global_frontier', PointStateSrv)
            frontier_point = Point(frontier_point[0], frontier_point[1], frontier_point[2])
            resp = point_state_srv(frontier_point)
            if resp.state.data: # still a global frontier
                return True
            return False
        except rospy.ServiceException:
            rospy.logwarn("is_global_frontier service call failed")
            return None


    def save_map(self, mapfiledir, mapfilename):
        # save map
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        filepath = mapfiledir+'/OccMap'
        if not isdir(filepath):
            mkdir(filepath)
        filepathname = filepath+'/'+mapfilename+'_'+timestr+'.ot'
        cmd = 'rosrun octomap_server octomap_saver ' + filepathname
        system(cmd)
        rospy.loginfo('Save map {}'.format(filepathname))

    def mapping(self):
        self.init_exploration()
        has_local_frontier = True
        # import ipdb; ipdb.set_trace()
        count = 0
        while not rospy.is_shutdown():
            count += 1
            # if self.args.global_only:
            #     if self.explore_global_frontier():
            #         time.sleep(2.0)
            #     else: # mapping finished
            #         break 
            # else: # A star planning on local map, move to nearest global frontier when no local frontiers
            #     if self.explore(): 
            #         pass
            #         # time.sleep(2.0)
            #     else: # no local frontier
            #         if self.explore_global_frontier():
            #             time.sleep(2.0)
            #         else:
            #             break 
            # if count%100==0:
            #     self.save_map(self.mapfiledir, self.mapfilename)
            # import ipdb;ipdb.set_trace()

            # # debug
            # A star planning on local map, move to nearest global frontier when no local frontiers
            if has_local_frontier:
                res = self.explore() 
                if not res:
                    has_local_frontier = False
            else: # no local frontier
                # import ipdb;ipdb.set_trace()
                res = self.teletrans_global_safepose()
                time.sleep(2.0)
                if res: 
                    has_local_frontier = True
                else:
                    break 


            if count%100==0:
                self.save_map(self.mapfiledir, self.mapfilename)
            # import ipdb;ipdb.set_trace()

        self.save_map(self.mapfiledir, self.mapfilename)

if __name__ == '__main__':
    rospy.init_node('expo_control', anonymous=True)
    args = get_args()

    controller = ExpoController(args)
    controller.mapping()

