#!/usr/bin/env python
import rospy
from roadmap_generator.srv import endpoints as EndPointsSrv
from roadmap_generator.srv import endpointsRequest, endpointsResponse
from roadmap_generator.srv import road as RoadSrv
from roadmap_generator.srv import roadRequest, roadResponse
from roadmap_generator.srv import roads as RoadsSrv
from roadmap_generator.srv import roadsRequest, roadsResponse
from roadmap_generator.srv import smooth as SmoothSrv
from roadmap_generator.srv import smoothRequest, smoothResponse
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, PoseArray

import time
import numpy as np

class NodePathSampler(object):
    '''
    This class defines interfaces for the ROS service
    This class doesn't store any data
    '''
    def __init__(self):
        # rospy.init_node('roadmap_path_sample', anonymous=True)


        self.node_vis = rospy.Publisher('sample_nodes', Marker, queue_size=1)
        self.path_vis = rospy.Publisher('ompl_path', Marker, queue_size=1)
        self.endpoints_vis = rospy.Publisher('end_points', Marker, queue_size=1)
        self.paths_vis = rospy.Publisher('ompl_paths', MarkerArray, queue_size=1, latch=True)

    def sample_nodes(self, nodenum, minx, maxx, miny, maxy, minz, maxz):
        '''
        Call sample_nodes_service, return list of Pose
        '''
        rospy.wait_for_service('sample_nodes_service')
        try:
            sample_nodes_srv = rospy.ServiceProxy('sample_nodes_service', EndPointsSrv)
            req = endpointsRequest()
            req.pointnum.data = nodenum
            req.range_minx.data = minx
            req.range_maxx.data = maxx
            req.range_miny.data = miny
            req.range_maxy.data = maxy
            req.range_minz.data = minz
            req.range_maxz.data = maxz
            res = sample_nodes_srv(req)
            assert isinstance(res, endpointsResponse)
        except rospy.ServiceException:
            rospy.logwarn("Endpoint service call failed!")
            return None

        if res.status.data:
            return res.nodes.poses
        else:
            rospy.logwarn("Endpoint service call: no nodes returned!")
            return None

    def plan_edge(self, init, goal):
        '''
        Call roadmap_srv, 
        Return list of Pose if there's a feasible path
        '''
        assert isinstance(init, Pose)
        assert isinstance(goal, Pose)
        rospy.wait_for_service('roadmap_srv')
        try:
            feasible_road = rospy.ServiceProxy('roadmap_srv', RoadSrv)
            req = roadRequest()
            req.init = init
            req.goal = goal
            res = feasible_road(req)
            assert isinstance(res, roadResponse)
            return res.roadmap.poses
        except rospy.ServiceException:
            rospy.logwarn("Planning failed")
            return None

    def plan_edges(self, init, goals):
        '''
        Call roadmap_srv, 
        Return list of Pose if there's a feasible path
        '''
        assert isinstance(init, Pose)
        goallist = PoseArray()
        goallist.poses=goals
        rospy.wait_for_service('plan_paths_srv')
        respathlist = []
        try:
            feasible_road = rospy.ServiceProxy('plan_paths_srv', RoadsSrv)
            req = roadsRequest()
            req.init = init
            req.goallist = goallist
            res = feasible_road(req)
            for path in res.roadmaplist:
                respathlist.append(path.poses)

            return respathlist
        except rospy.ServiceException:
            rospy.logwarn("Planning failed")
            return None

    def smooth_path(self, path):
        '''
        Call path_smooth_service
        Return list of smoothed poses 
        '''
        rospy.wait_for_service('path_smooth_service')
        try:
            smooth_path_srv = rospy.ServiceProxy('path_smooth_service', SmoothSrv)
            req = smoothRequest()
            req.roadmap.poses = path
            # import ipdb;ipdb.set_trace()
            res = smooth_path_srv(req)
            return res.smooth_roadmap.poses
        except rospy.ServiceException:
            rospy.logwarn("Smooth failed")
            return None

    def vis_graph(self, graph, visedge = True):
        poses = []
        for node in graph.nodes:
            poses.append(node)
        self.publish_nodes_marker(poses)

        if visedge:
            for k,edge in enumerate(graph.edges):
                self.publish_path_marker(graph.get_edge_data(edge[0],edge[1])['path'], k)

    def publish_nodes_marker(self, poses, scale=0.5, alpha=1.0, r = 1.0, g=1.0, b=0.0, id_ = 0):
        markers = Marker()
        markers.header.frame_id = 'map'
        markers.header.stamp = rospy.Time.now()
        markers.type = Marker.CUBE_LIST
        markers.action = Marker.ADD
        markers.ns = 'nodes'
        markers.id = id_
        markers.color.r = r
        markers.color.g = g
        markers.color.b = b
        markers.color.a = alpha
        markers.scale.x = scale
        markers.scale.y = scale
        markers.scale.z = scale
        for k,point in enumerate(poses): # isinstance(point, Pose)
            node_disp = Point(point.position.x, point.position.y, point.position.z)
            markers.points.append(node_disp)
        # import ipdb;ipdb.set_trace()
        time.sleep(0.01)
        self.node_vis.publish(markers)

    def delete_nodes_marker(self, id_ = 0):
        markers = Marker()
        markers.type = Marker.CUBE_LIST
        markers.action = Marker.DELETE
        markers.ns = 'nodes'
        markers.id = id_
        time.sleep(0.01)
        self.node_vis.publish(markers)

    def get_path_marker(self, path, markerid=0, hightlight=True):
        markers = Marker()
        markers.header.frame_id = 'map'
        markers.header.stamp = rospy.Time.now()
        markers.type = Marker.LINE_STRIP
        markers.action = Marker.ADD
        markers.ns = 'edge'
        markers.id = markerid 
        if hightlight:
            r, g, b, a, scale = 0.2, 0.2, 1.0, 1.0, 0.3
        else:
            r, g, b, a, scale = 0.5, 0.5, 0.7, 0.8, 0.1
        markers.color.r = r #+ markerid*0.1
        markers.color.g = g #+ markerid*0.1
        markers.color.b = b 
        markers.color.a = a
        markers.scale.x = scale
        markers.scale.y = scale
        markers.scale.z = scale
        markers.pose.orientation.w = 1.
        for k,point in enumerate(path):
            node_disp = Point(point.position.x, point.position.y, point.position.z)
            # import ipdb;ipdb.set_trace()
            markers.points.append(node_disp)
        return markers
    
    def delete_path_marker(self, markerid=0):
        markers = Marker()
        markers.header.frame_id = 'map'
        markers.header.stamp = rospy.Time.now()
        markers.type = Marker.LINE_STRIP
        markers.action = Marker.DELETE
        markers.ns = 'edge'
        markers.id = markerid
        return markers


    def publish_path_marker(self, path, markerid=0):
        markers = self.get_path_marker(path, markerid)
        time.sleep(0.01)
        self.path_vis.publish(markers)

    def publish_paths_marker(self, paths, hightlight=True, deletepaths=False):
        markers = MarkerArray()
        for k, path in enumerate(paths):
            if not deletepaths:
                marker = self.get_path_marker(path, markerid=k, hightlight=hightlight)
                markers.markers.append(marker)
            else:
                marker = self.delete_path_marker(markerid=k)
                markers.markers.append(marker)
        time.sleep(0.01)
        if len(markers.markers) > 0:
            self.paths_vis.publish(markers)

    def publish_endpoints_marker(self, init, goal):
        markers = Marker()
        markers.header.frame_id = 'map'
        markers.header.stamp = rospy.Time.now()
        markers.type = Marker.ARROW
        markers.action = Marker.ADD
        markers.ns = 'map'
        markers.id = 2
        markers.color.r = 0.7
        markers.color.g = 0.2
        markers.color.b = 0.9
        markers.color.a = 1.0
        markers.scale.x = 1.0
        markers.scale.y = 1.0
        markers.scale.z = 1.0
        markers.points.append(Point(init.position.x, init.position.y, init.position.z))
        markers.points.append(Point(goal.position.x, goal.position.y, goal.position.z))
        # import ipdb;ipdb.set_trace()
        self.endpoints_vis.publish(markers)