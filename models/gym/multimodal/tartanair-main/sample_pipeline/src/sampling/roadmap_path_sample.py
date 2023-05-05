#!/usr/bin/env python
import sys
import os
import yaml
curdir = os.path.dirname(os.path.realpath(__file__))
print(curdir)
sys.path.insert(0,curdir+'/..')
sys.path.insert(0,curdir)
# import sys
# sys.path.append('../')

import time
import numpy as np
import rospy
import matplotlib.pyplot as plt
from os import mkdir, listdir, system
from os.path import exists, join, isfile, isdir, dirname, realpath

import pickle
from RandomPoseReSampler import RandomwalkPoseReSampler
from RandomOrientationSampler import QuaternionSampler
from NodePathSampler import NodePathSampler
from PathGraph import PathGraph

from settings import get_args
import copy

# from settings import get_args
from data_collection_panel.srv import LoadGraph, LoadGraphResponse
from data_collection_panel.srv import SampleNode, SampleNodeResponse
from data_collection_panel.srv import PanelSimpleService, PanelSimpleServiceResponse
from data_collection_panel.srv import SampleGraph, SampleGraphResponse
from data_collection_panel.srv import SamplePath, SamplePathResponse
from data_collection_panel.srv import GeneratePose, GeneratePoseResponse
from data_collection_panel.srv import SavePath, SavePathResponse
from data_collection_panel.srv import LoadPaths, LoadPathsResponse


class RoadmapPathSample(object):
    """docstring for RoadmapPathSample"""
    def __init__(self):
        # self.rate =None

        self.args = get_args()
        self.MaxFailureNum = self.args.max_failure_num

        self.sample_node_flag = False
        self.sample_graph_flag = False
        self.sample_path_flag = False
        self.sample_path_wait_flag = False
        # self.sample_position_flag = False
        self.save_path_flag=0 
        # self.sampling_node_flag = False # True is sample_node_button is pressed even once 


        # variables for sampling
        self.OutputDir = ""
        self.GraphFilename = ""
        self.PathOutdir = ""
        self.save_path_ind = 0

        self.SamplePath = False # loopy path
        self.SamplePosition = False # random linear velociy 
        # self.PruneGraphNodeNum = False

        self.positions_np = None
        self.pathsamplenum = 0

        self.Nodenum = 0 ###
        self.Edgenum = 0 ###
        self.TrajlenMinThresh = 0 ####
        self.TrajlenMaxThresh = 0 ####
        self.MinX = 0 ###
        self.MaxX = 0 ###
        self.MinY = 0 ###
        self.MaxY = 0 ###
        self.MinZ = 0 ###
        self.MaxZ = 0 ###

        self.reset_graph()
        self.network_bk = None # for reload the graph

        # for visualizing the paths
        self.planpathlist = []
        self.pathlist = []
        self.vis_path_ind = -1

        self.dir_path = dirname(realpath(__file__))


    def reset_graph(self):
        self.pathgraph = PathGraph()
        self.pathsampler = NodePathSampler()###
        self.nodeposes_temp = [] # store temporary nodes during node-sampling
        # self.nodeposes = []
        # self.nodeslinknum = np.empty([0])
        self.pathsampler.delete_nodes_marker(id_=0)
        self.pathsampler.delete_nodes_marker(id_=1)
        self.sample_path_flag = False


    def node_distance(self,init, goal):
        diff = np.array([goal.position.x-init.position.x,
                         goal.position.y-init.position.y,
                         goal.position.z-init.position.z])
        return np.linalg.norm(diff, axis=0)

    def node_in_range(self, node, minX, maxX, minY, maxY, minZ, maxZ):
        if node.position.x > minX and \
            node.position.x < maxX and \
            node.position.y > minY and \
            node.position.y < maxY and \
            node.position.z > minZ and \
            node.position.z < maxZ:
            return True
        return False

    def load_graph_cb(self, request):
        self.reset_graph()
        graphfilename = request.graph_filename
        response = LoadGraphResponse()
        if isfile(graphfilename): # load graph file from file
            self.pathgraph = PathGraph(graphfilename) 

            nodenum = self.pathgraph.get_nodenum()
            edgenum = self.pathgraph.get_edgenum()
            # self.nodeposes = []
            # self.nodeslinknum = np.zeros(nodenum, dtype=np.int32)
            # for k, node in enumerate(self.pathgraph.graph.nodes):
            #     self.nodeposes.append(node)
            #     edgenum = len(self.pathgraph.graph.edges(node))
            #     self.nodeslinknum[k] = edgenum
            self.pathsampler.vis_graph(self.pathgraph.graph, visedge=False)
            rospy.loginfo("Load graph file {}, {} nodes, {} edges..".format(graphfilename, nodenum, edgenum))
            response.success_flag = True
            return response

        else:
            rospy.logwarn("Cannot find the graph file: {}".format(graphfilename))
            response.success_flag = False
            return response

    def reload_grpah_cb(self, request):
        self.reset_graph()
        response = PanelSimpleServiceResponse()
        if self.network_bk is not None:
            self.pathgraph.graph = self.network_bk
            self.pathsampler.vis_graph(self.pathgraph.graph, visedge=False)
            nodenum = self.pathgraph.get_nodenum()
            edgenum = self.pathgraph.get_edgenum()
            rospy.loginfo("Reload from recent graph {} nodes, {} edges..".format(nodenum, edgenum))
            response.success_flag = True
        else:
            response.success_flag = False
        return response

    def load_nodes_from_graph_cb(self, request):
        self.reset_graph()
        graphfilename = request.graph_filename
        response = LoadGraphResponse()
        if isfile(graphfilename): # load graph file from file
            self.pathgraph = PathGraph(graphfilename, clear_edges=True) 

            nodenum = self.pathgraph.get_nodenum()
            edgenum = self.pathgraph.get_edgenum()
            # self.nodeposes = []
            # self.nodeslinknum = np.zeros(nodenum, dtype=np.int32)
            # for k, node in enumerate(self.pathgraph.graph.nodes):
            #     self.nodeposes.append(node)
            #     edgenum = len(self.pathgraph.graph.edges(node))
            #     self.nodeslinknum[k] = edgenum
            self.pathsampler.vis_graph(self.pathgraph.graph, visedge=False)
            rospy.loginfo("Load nodes from graph file {}, {} nodes, {} edges..".format(graphfilename, nodenum, edgenum))
            response.success_flag = True
            return response

        else:
            rospy.logwarn("Cannot find the graph file: {}".format(graphfilename))
            response.success_flag = False
            return response

    def sample_node_cb(self,request):
        rospy.loginfo("SampleNode called")
        ###################################
        # read args
        self.Nodenum = request.node_num ###
        self.MinX = min(request.node_range_xmin, request.node_range_xmax)
        self.MaxX = max(request.node_range_xmin, request.node_range_xmax)
        self.MinY = min(request.node_range_ymin, request.node_range_ymax) 
        self.MaxY = max(request.node_range_ymin, request.node_range_ymax)
        self.MinZ = min(request.node_range_zmin, request.node_range_zmax)
        self.MaxZ = max(request.node_range_zmin, request.node_range_zmax)

        ## Return statment
        response = SampleNodeResponse()
        response.success_flag = True

        self.sample_node_flag = True

        return response
        ###################################

    def sample_node_func(self):
        if self.sample_node_flag:
            # node sampling through ros service
            self.nodeposes_temp = self.pathsampler.sample_nodes(self.Nodenum, self.MinX, self.MaxX, 
                                                                self.MinY, self.MaxY, self.MinZ, self.MaxZ)

            self.pathsampler.publish_nodes_marker(self.pathgraph.get_node_list(), scale=0.5, alpha=1.0)
            if self.nodeposes_temp is not None:
                self.pathsampler.publish_nodes_marker(self.nodeposes_temp, scale=0.52, alpha =1.0, r = 0.0, g=0.0, b=1.0, id_=1)
            else:
                rospy.logwarn('No nodes returned by the sample nodes service. ')
            rospy.loginfo("sample_node_func call ended")
            self.sample_node_flag = False
            # self.sampling_node_flag = True # used to track the first button press on sample_path_button

    def add_node_cb(self,request):
        # combine the pre-loaded nodes and link 
        if len(self.nodeposes_temp)>0:
            self.pathgraph.add_nodes(self.nodeposes_temp)
            # self.nodeposes = self.nodeposes + self.nodeposes_temp 
            # nodeslinknum_temp = np.zeros(len(self.nodeposes_temp), dtype=np.int32)
            # self.nodeslinknum = np.concatenate((self.nodeslinknum, nodeslinknum_temp))
            # self.Nodenum = len(self.nodeposes)
            nodenum = self.pathgraph.get_nodenum()
            print('Added new {} nodes into the nodelist, total nodes {}'.format(len(self.nodeposes_temp), nodenum))
            self.nodeposes_temp = []
            self.pathsampler.publish_nodes_marker(self.pathgraph.get_node_list(), scale=0.5, alpha=1.0)
            self.pathsampler.delete_nodes_marker(id_=1)

        response = PanelSimpleServiceResponse()
        response.success_flag = True
        return response

    def delete_node_cb(self, request):
        rospy.loginfo("DeleteNode called")
        ###################################
        # read args
        minX = min(request.node_range_xmin, request.node_range_xmax)
        maxX = max(request.node_range_xmin, request.node_range_xmax)
        minY = min(request.node_range_ymin, request.node_range_ymax) 
        maxY = max(request.node_range_ymin, request.node_range_ymax)
        minZ = min(request.node_range_zmin, request.node_range_zmax)
        maxZ = max(request.node_range_zmin, request.node_range_zmax)

        nodelist = self.pathgraph.get_node_list() 
        self.nodeposes_temp = []
        for node in nodelist:
            if self.node_in_range(node, minX, maxX, minY, maxY, minZ, maxZ):
                self.nodeposes_temp.append(node)

        self.pathgraph.delete_nodes(self.nodeposes_temp)
        rospy.loginfo("Delete {} nodes in the range, {} nodes remain..".format(len(self.nodeposes_temp), self.pathgraph.get_nodenum()))
        self.pathsampler.publish_nodes_marker(self.pathgraph.get_node_list(), scale=0.5, alpha=1.0)
        self.pathsampler.publish_nodes_marker(self.nodeposes_temp, scale=0.55, alpha=1.0, r = 0.0, g=0.0, b=1.0, id_ = 1)

        ## Return statment
        response = SampleNodeResponse()
        response.success_flag = True

        return response

    def save_output_dir(self, output_dir):
        config_file = self.dir_path.split('sample_pipeline')[0]+'/data_collection_panel/resource/.output_dir'
        with open(config_file, 'w') as f:
            f.write(output_dir)
        rospy.loginfo('Save {} to the config file {}'.format(output_dir, config_file))

    def sample_graph_cb(self,request):
        self.OutputDir = request.output_dir
        self.Edgenum = request.edge_num ###
        self.TrajlenMinThresh = request.min_dist_thresh ####
        self.TrajlenMaxThresh = request.max_dist_thresh ####

        self.pathsampler.publish_nodes_marker(self.pathgraph.get_node_list())
        self.pathsampler.delete_nodes_marker(id_=1)

        response = SampleGraphResponse()
        response.success_flag = True

        # prepare the output folder
        if not isdir(self.OutputDir):
            try:
                mkdir(self.OutputDir)
            except: # output folder is wrong
                response.success_flag = False
                return response

        self.save_output_dir(self.OutputDir)
        self.sample_graph_flag = True
        return response


    def sample_graph_func(self, parallelnum=10):
        if self.sample_graph_flag:
            nodelist = self.pathgraph.get_node_list()
            nodenum = self.pathgraph.get_nodenum()
            edgenum = self.pathgraph.get_edgenum()
            rospy.loginfo("Start to sample edge for {} nodes, {} edges already exists..".format(nodenum, edgenum))
            for k, initpose in enumerate(nodelist): # add link to the nodes
                # initpose = self.nodeposes[k]
                randlist_after = np.random.permutation(nodenum - k -1) + k + 1 # query nodes after the current one first
                randlist_befor = np.random.permutation(k)
                randlist = np.concatenate((randlist_after, randlist_befor))
                ind = 0
                failnum = 0
                initpose_nodenum = self.pathgraph.get_node_edgenum(initpose)
                while initpose_nodenum < self.Edgenum: # randomly sample from all the nodes
                    if ind >= nodenum-1: # Not enough edges after all the nodes have been sampled
                        print('No enough edges '+str(initpose_nodenum) + ' sampled')
                        break
                    goallist = [] # for parrellization
                    indlist = []
                    while len(goallist) < parallelnum:
                        if ind >= nodenum-1: 
                            break
                        randind = randlist[ind]
                        ind += 1
                        goalpose = nodelist[randind]
                        if self.pathgraph.has_edge(initpose, goalpose):
                            print('edge exists ({}, {})'.format(k,randind))
                            continue
                        nodedist = self.node_distance(initpose, goalpose)
                        if nodedist < self.TrajlenMinThresh or nodedist > self.TrajlenMaxThresh: # do not link to nearby node
                            continue
                        goallist.append(goalpose)
                        indlist.append(randind)
                    if len(goallist)>0: 
                        paths = self.pathsampler.plan_edges(initpose, goallist)
                        for w, path in enumerate(paths):
                            if (path is not None) and (len(path)>0): 
                                self.pathgraph.graph.add_edge(initpose, goallist[w], path=path)
                                initpose_nodenum = self.pathgraph.get_node_edgenum(initpose)
                                print('{} - {}, find {} edges'.format(k, indlist[w], initpose_nodenum))
                                # self.pathsampler.publish_path_marker(path)
                            else: # planning failed
                                failnum += 1
                                if failnum>self.MaxFailureNum:
                                    print('Too many failures, got edges of '+str(initpose_nodenum))
                                    break
                        self.pathsampler.publish_paths_marker(paths)
                        self.planpathlist = paths
                    # import ipdb;ipdb.set_trace()
                    if not self.sample_graph_flag:
                        rospy.loginfo('Sample graph paused..')
                        break
                if not self.sample_graph_flag:
                    break

            # import ipdb;ipdb.set_trace()
            # save the graph to disk
            outputgraphfilename = 'node{}_edge{}_len{}_{}.graph'.format(self.pathgraph.get_nodenum(), self.pathgraph.get_edgenum(), 
                                                                        self.TrajlenMinThresh, self.TrajlenMaxThresh)
            outputgraphfiledir = self.OutputDir + '/OccMap/'
            if not isdir(outputgraphfiledir):
                mkdir(outputgraphfiledir)
            outputgraphfilename = outputgraphfiledir + outputgraphfilename
            if isfile(outputgraphfilename):
                timestr = time.strftime('%m%d_%H%M%S',time.localtime())
                outputgraphfilename = outputgraphfilename.split('.graph')[0] + '_' + timestr+'.graph'
            pickle.dump(self.pathgraph.graph, open(outputgraphfilename, 'wb'))
            rospy.loginfo('Graph file saved '+ outputgraphfilename)

            # save a txt file for all the nodes
            poselist = self.pathgraph.get_node_list()
            positionlist = self.pathgraph.poselist2positionlist(poselist)
            outputnodefilename = outputgraphfilename.split('.graph')[0] + '_nodes.txt'
            np.savetxt(outputnodefilename, np.array(positionlist))
            rospy.loginfo('Nodes file saved '+ outputnodefilename)

            self.sample_graph_flag = False

    def sample_graph_pause_cb(self, request):
        rospy.loginfo('Pause graph sample.. ')
        self.sample_graph_flag = False
        response = PanelSimpleServiceResponse()
        response.success_flag = True
        return response

    def check_org_path_folder(self, pathfolder):
        # find the paths in the folder
        existing_files = listdir(pathfolder)
        existing_files = [ef for ef in existing_files if ef.endswith('.txt')]
        existing_files.sort()
        rospy.loginfo("Find {} path files in folder {}".format(len(existing_files), self.PathOutdir))
        # organize the paths, and name after them
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        for k, ff in enumerate(existing_files):
            fname = ff.split('_'+timestr)[0] + '.txt'
            fname_temp = 'P'+str(k).zfill(3)+'_'+timestr+'.txt'
            if ff != fname: # rename the file
                cmd = 'mv ' + pathfolder + '/' + ff + ' ' + pathfolder + '/' + fname_temp
                system(cmd) # rename the file to a temp name to avoid collision
        # remove the temp str in filenames
        existing_files = listdir(pathfolder)
        existing_files = [ef for ef in existing_files if ef.endswith('.txt')]
        for k, ff in enumerate(existing_files):
            if ff.split('.txt')[0].endswith(timestr):
                fname = ff.split('_'+timestr)[0] + '.txt'
                cmd = 'mv ' + pathfolder + '/' + ff + ' ' + pathfolder + '/' + fname
                system(cmd)

        return len(existing_files) 
        
    def sample_path_cb(self,request):
        self.OutputDir = request.output_dir
        self.ros_path_folder = request.ros_path_folder

        self.SampleCycleMode = request.sample_cycle_mode
        self.cycle_min_nodes = request.cycle_min_nodes 
        self.cycle_max_nodes = request.cycle_max_nodes # TODO: implement the min/max requirement 
        rospy.loginfo("Sample path called: minnodes {}, maxnodes {}".format(self.cycle_min_nodes, self.cycle_max_nodes))
        # self.interactive = request.interactive

        if not self.sample_path_flag: # first time being called on the current graph, 
            self.network_bk = copy.deepcopy(self.pathgraph.graph) # make a backup of the current graph
            # do some initialization
            self.PathOutdir = self.OutputDir + '/' + self.ros_path_folder
            self.save_path_ind = 0
            self.pathsamplenum = 0
            if isdir(self.PathOutdir): # not to overwrite the existing path
                self.save_path_ind = self.check_org_path_folder(self.PathOutdir)
            else:
                mkdir(self.PathOutdir)
            rospy.loginfo("Path file will be saved in {}, starting from index {}".format(self.PathOutdir, self.save_path_ind))
            self.sample_path_flag = True
            self.save_output_dir(self.OutputDir)

            # clear the visualization of the planed path
            if len(self.planpathlist)>0:
                self.pathsampler.publish_paths_marker(self.planpathlist,deletepaths=True)

        # save the output-dir into the hidder file
        self.save_output_dir(self.OutputDir)
        # sample loop from the graph
        response = SamplePathResponse()
        if self.sample_loop(): # new loop sampled
            response.success_flag = True

        else: # no loop can be sampled on the current graph
            self.sample_path_flag = False
            response.success_flag = False

        return response


    def sample_loop(self):

        self.pathsampler.vis_graph(self.pathgraph.graph, visedge=False)
        cycle = self.pathgraph.sample_cycle(mode=self.SampleCycleMode, minnodes=self.cycle_min_nodes, maxnodes=self.cycle_max_nodes)
        if cycle is None: # no more cycle
            self.sample_path_flag = False
            self.positions_np = None
            return False
        cycle_poses = self.pathgraph.cycle2poselist(cycle)
        self.pathsamplenum += 1
        rospy.loginfo('{} - cycle nodes: {}, path length: {}'.format(self.pathsamplenum, len(cycle), len(cycle_poses)))

        smooth_cycle_poses = self.pathsampler.smooth_path(cycle_poses)
        rospy.loginfo ('length after smooth: {}'.format(len(smooth_cycle_poses)))
        positionlist = self.pathgraph.poselist2positionlist(smooth_cycle_poses)
        self.positions_np = np.array(positionlist, dtype=np.float32)

        # visuualization
        self.pathsampler.publish_path_marker(smooth_cycle_poses,markerid=1) 
        self.pathsampler.publish_nodes_marker(cycle, scale=0.55, alpha=1.0, r = 0.0, g=0.0, b=1.0, id_ = 1)

        self.pathgraph.delete_nodes(cycle)
        return True


    def save_path_cb(self,request):
        if self.positions_np is not None:
            pathfilename = self.PathOutdir+'/P'+str(self.save_path_ind).zfill(3)
            self.save_path_ind += 1
            np.savetxt(pathfilename+'.txt', self.positions_np)
            # self.pathgraph.visualize_poslist(self.positions_np, pathfilename+'.png')
            self.positions_np = None
            rospy.loginfo("Path saved {}".format(pathfilename) )

        response = SavePathResponse()
        response.success_flag = True
        return response


    def load_paths_cb(self, request):
        outputDir = request.output_dir
        ros_path_folder = request.ros_path_folder
        pathdir = outputDir + '/' + ros_path_folder
        self.pathlist = [] # this will be ussed in the next_path_cb and prev_path_cb
        self.vis_path_ind = -1
        if isdir(pathdir): # load path from this directory
            pathnum = self.check_org_path_folder(pathdir)
            pathfiles = listdir(pathdir)
            pathfiles = [ef for ef in pathfiles if ef.endswith('.txt')]
            pathfiles.sort()
            for pathfile in pathfiles:
                path = np.loadtxt(pathdir + '/' + pathfile)
                poselist = self.pathgraph.positionlist2poselist(path)
                self.pathlist.append(poselist)

            rospy.loginfo("Path file loaded from {}, path number {}".format(pathdir, len(self.pathlist)))
        self.pathsampler.publish_paths_marker(self.pathlist, hightlight=True)

        # save the output-dir into the hidder file
        self.OutputDir = outputDir
        self.save_output_dir(self.OutputDir)
        response = LoadPathsResponse()
        response.success_flag = True
        return response

    def path_statistics(self, path):
        '''
        path is a list of pose
        '''
        poselist = []
        for pose in path:
            poselist.append([pose.position.x, pose.position.y, pose.position.z])
        poses = np.array(poselist)
        rospy.loginfo("  Path-x min {:.2f}, max {:.2f}, std {:.2f}".format(poses[:,0].min(), poses[:,0].max(), poses[:,0].std()))
        rospy.loginfo("  Path-y min {:.2f}, max {:.2f}, std {:.2f}".format(poses[:,1].min(), poses[:,1].max(), poses[:,1].std()))
        rospy.loginfo("  Path-z min {:.2f}, max {:.2f}, std {:.2f}".format(poses[:,2].min(), poses[:,2].max(), poses[:,2].std()))

    def next_path_cb(self, request):
        pathnum = len(self.pathlist)
        if pathnum>0: 
            self.vis_path_ind = (self.vis_path_ind + 1)%pathnum
            self.pathsampler.publish_paths_marker(self.pathlist, hightlight=False)
            self.pathsampler.publish_path_marker(self.pathlist[self.vis_path_ind])
            rospy.loginfo("Visualizing path number {}, path length {}".format(self.vis_path_ind, len(self.pathlist[self.vis_path_ind])))
            self.path_statistics(self.pathlist[self.vis_path_ind])

        response = PanelSimpleServiceResponse()
        response.success_flag = True
        return response

    def prev_path_cb(self, request):
        pathnum = len(self.pathlist)
        if pathnum>0: 
            self.vis_path_ind = self.vis_path_ind - 1
            if self.vis_path_ind < 0:
                self.vis_path_ind = pathnum-1
            self.pathsampler.publish_paths_marker(self.pathlist, hightlight=False)
            self.pathsampler.publish_path_marker(self.pathlist[self.vis_path_ind])
            rospy.loginfo("Visualizing path number {}".format(self.vis_path_ind))

        response = PanelSimpleServiceResponse()
        response.success_flag = True
        return response


    def sample_position_cb(self, request):
        MaxVel = request.vel_max
        MinVel = request.vel_min
        MaxAcc = request.acc_max
        RandDegree = request.rand_degree
        SmoothCount = request.smooth_count
        MaxYaw = request.yaw_max
        MinYaw = request.yaw_min
        MaxPitch = request.pitch_max 
        MinPitch = request.pitch_min
        MaxRoll = request.roll_max 
        MinRoll = request.roll_min
        OutFolder = request.output_folder
        EnvDir = request.data_dir
        InputFolder = request.input_folder

        inputfolder = join(EnvDir, InputFolder)
        outputfolder = join(EnvDir, OutFolder)

        if not exists(outputfolder):
            mkdir(outputfolder)
        else: # folder exist, make a new folder
            timestr = time.strftime('%m%d_%H%M%S',time.localtime())
            outputfolder = outputfolder + '_' + timestr
            print("Output folder exists! Output to new folder: {}".format(outputfolder))
            mkdir(outputfolder)

        print("Sample pose called, inputdir {}, outputdir {}".format(inputfolder, outputfolder))

        config = {'vel_max': request.vel_max, 
                  'vel_min': request.vel_min,
                  'acc_max': request.acc_max,
                  'rand_degree': request.rand_degree,
                  'smooth_count': request.smooth_count,
                  'yaw_max': request.yaw_max,
                  'yaw_min': request.yaw_min,
                  'pitch_max': request.pitch_max, 
                  'pitch_min': request.pitch_min,
                  'roll_max': request.roll_max, 
                  'roll_min': request.roll_min,}
        with open(join(outputfolder, 'pose_config.yaml'), 'w') as f:
            yaml.dump(config, f)

        posesampler = RandomwalkPoseReSampler(max_vel = MaxVel, min_vel=MinVel, camera_rate=10.0, max_acc = MaxAcc)
        orientationsampler = QuaternionSampler(RandDegree, SmoothCount, MaxYaw, MinYaw, MaxPitch, MinPitch, MaxRoll, MinRoll)
        suc = self.rospath2poselist(posesampler, orientationsampler, inputfolder, outputfolder)

        response = GeneratePoseResponse()
        response.success_flag = suc

        return response

    def rospath2poselist(self, posesampler, orientationsampler, trajdir, outdir):

        trajfiles = listdir(trajdir)
        trajfiles = [ff for ff in trajfiles if ff[-3:]=='txt']
        trajfiles.sort()
        print("Sample Pose: find trajections {}, in {}".format(trajfiles, trajdir))

        for k,trajfile in enumerate(trajfiles):
            print("Trajecotry {}: {}".format(k, trajfile))
            posefilename = outdir + '/Pose_' + (str(k).zfill(3))  + '.txt'
            positionfilename = outdir + '/Position_' + str(k).zfill(3) + '.txt'
            positionfigname = outdir + '/Position_' + str(k).zfill(3) + '.png'
            orientationfigname = outdir + '/Orientation_' + str(k).zfill(3) + '.png'
            traj_np = np.loadtxt(trajdir + '/' + trajfile)
            positions = posesampler.sample_poses(traj_np.tolist(), visfilename=positionfigname)
            if positions is None:
                print("***  Sample position failed..  ***")
                return False
            np.savetxt(positionfilename, positions)       
            anglelist, quatlist = orientationsampler.sample_orientations(len(positions), orientationfigname)
            poses = np.concatenate((positions, np.array(quatlist)), axis=1)
            np.savetxt(posefilename, poses)
        print("**********************************")
        print("*     Sample Completed !         *")
        print("**********************************")
        return True
        
def main():
    rospy.init_node("sample_nodes_roadmap", anonymous=True)

    roadmapPathSample_object = RoadmapPathSample()

    rospy.loginfo("sample_nodes_roadmap started")

    rate = rospy.Rate(20)
    roadmapPathSample_object.rate = rate

    load_graph_srv = rospy.Service('load_graph',LoadGraph,roadmapPathSample_object.load_graph_cb)
    reload_graph_srv = rospy.Service('reload_graph',PanelSimpleService,roadmapPathSample_object.reload_grpah_cb)
    load_nodes_srv = rospy.Service('load_nodes_from_graph',LoadGraph,roadmapPathSample_object.load_nodes_from_graph_cb)
    
    sample_node_srv = rospy.Service('sample_node',SampleNode,roadmapPathSample_object.sample_node_cb)
    add_node_srv = rospy.Service('add_node',PanelSimpleService,roadmapPathSample_object.add_node_cb)
    delete_node_srv = rospy.Service('delete_node',SampleNode,roadmapPathSample_object.delete_node_cb)
    sample_graph_srv = rospy.Service('sample_graph',SampleGraph,roadmapPathSample_object.sample_graph_cb)
    sample_graph_pause_srv = rospy.Service('sample_graph_pause',PanelSimpleService,roadmapPathSample_object.sample_graph_pause_cb)

    sample_path_srv = rospy.Service('sample_path',SamplePath,roadmapPathSample_object.sample_path_cb)
    generate_pose_srv = rospy.Service('generate_pose',GeneratePose,roadmapPathSample_object.sample_position_cb)
    save_path_srv = rospy.Service('save_path',SavePath,roadmapPathSample_object.save_path_cb)

    load_paths_srv = rospy.Service('load_paths',LoadPaths,roadmapPathSample_object.load_paths_cb)
    next_path_srv = rospy.Service('next_path',PanelSimpleService,roadmapPathSample_object.next_path_cb)
    prev_path_srv = rospy.Service('prev_path',PanelSimpleService,roadmapPathSample_object.prev_path_cb)
    
    while not rospy.is_shutdown():
        roadmapPathSample_object.sample_node_func()
        roadmapPathSample_object.sample_graph_func()
        # roadmapPathSample_object.sample_path_func()
        # roadmapPathSample_object.sample_position_func()
        rate.sleep()
    rospy.spin()
          

if __name__ == '__main__':
    main()

    # rospy.init_node("sample_nodes_roadmap", anonymous=True)
    # roadmapPathSample_object = RoadmapPathSample()
    # roadmapPathSample_object.sample_position_cb(None)

































