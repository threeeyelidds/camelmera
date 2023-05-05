'''
1. Do statistics on depth
2. Do statistics on rgb
3. Verify the segmentation
4. Output video
'''
import os
import sys
from wsgiref.util import FileWrapper
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

import numpy as np
import cv2
from os.path import isfile, join, isdir, split, splitext
from os import listdir, mkdir, environ
import time
import yaml
from multiprocessing import Pool

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in environ ) ):
    plt.switch_backend('agg')
    # These print() calls cluter the terminal when used on Windows.
    # print("Environment variable DISPLAY is not present in the system.")
    # print("Switch the backend of matplotlib to agg.")

from .data_verification import verify_traj_depth, verify_traj_rgb, verify_traj_seg, get_seg_color_label_value
from .data_visualization import save_vid_with_statistics, vis_frame, DataVisualizer
from .data_enumeration import enumerate_frames, enumerate_modalities
from .ImageReader import ImageReader

def get_filename_parts(fn):
    s0 = split(fn)
    s1 = splitext( s0[1] )
    d = '.' if s0[0] == '' else s0[0]
    return d, *s1

class FileLogger():
    def __init__(self, filename, overwrite=False):
        if isfile(filename):
            if overwrite:
                print('Overwrite existing file {}'.format(filename))
            else:
                timestr = time.strftime('%m%d_%H%M%S',time.localtime())
                filename = filename+'_'+timestr
        self.filename = filename
        self.f = open(filename, 'w')

    def log(self, logstr):
        self.f.write(logstr)

    def logline(self, logstr):
        self.f.write(logstr+'\n')

    def close(self,):
        self.f.close()

class SegErrorFileWriter(FileLogger):
    def dump(self,errordict):
        # write the bad frames to a txt file
        for key, values in errordict.items(): 
            self.logline('%s %d' % (key, len(values)))
            for v in values:
                self.logline('  %s' % (v))

class SegErrorFileReader():
    def __init__(self, filename):
        if not isfile(filename):
            print('Error File not Found {}'.format(filename))
            self.f = None
        self.filename = filename
        self.f = open(filename, 'r')

    def load(self):
        if self.f is None:
            print("Error loading Seg file: {}".format(self.filename))
            return {}
        lines = self.f.readlines()
        errordict = {}
        ind = 0
        while ind<len(lines):
            line = lines[ind].strip()
            key, valuelen = line.split(' ')
            key = int(key)
            valuelen = int(valuelen)
            errorlist = []
            ind += 1
            for k in range(valuelen):
                if ind >= len(lines):
                    print("Seg Error File Load Error: {}, line {}".format(self.filename, ind))
                    return {}
                line = lines[ind].strip()
                errorlist.append(line)
                ind += 1
            errordict[key] = errorlist
        self.f.close()
        return errordict


def plot_depth_info(dispmax, dispmin, dispmean, dispstd, disphistfig, binnum=500):
    # save depth statistic figures
    plt.figure(figsize=(12,12))
    showlog = True
    plt.subplot(4,1,1)
    plt.hist(dispmean.reshape(-1),bins=binnum, log=showlog)
    plt.title('disp mean')
    plt.grid()

    plt.subplot(4,1,2)
    plt.hist(dispstd.reshape(-1),bins=binnum, log=showlog)
    plt.title('disp std')
    plt.grid()

    plt.subplot(4,1,3)
    plt.hist(dispmax.reshape(-1),bins=binnum, log=showlog)
    plt.title('disp max')
    plt.grid()

    plt.subplot(4,1,4)
    plt.hist(dispmin.reshape(-1),bins=binnum, log=showlog)
    plt.title('disp min')
    plt.grid()

    # plt.show()
    plt.savefig(disphistfig)
    plt.close()

def plot_rgb_info(rgbmean, rgbstd, rgbfig_traj):
    rgbmean12cam = rgbmean.mean(axis=1)
    rgbstd12cam = rgbstd.mean(axis=1)
    plt.figure(figsize=(12,6))
    plt.plot(np.array(rgbmean12cam))
    plt.plot(np.array(rgbstd12cam))
    plt.legend(['mean', 'std'])
    plt.title('RGB Mean and Std')
    plt.grid()
    plt.savefig(rgbfig_traj)
    plt.close()

def save_preview_video(env_root_dir, traj_list, vid_out_dir = 'video'):
    '''
    Input: Trajectory folder is organized in <env_root_dir>/data_folder[k]/P00X
    Output: a video for each trajectory: <vid_out_dir>/<datafolder>_P00X.mp4
    '''
    vidoutdir = join(env_root_dir, vid_out_dir)
    if not isdir(vidoutdir):
        mkdir(vidoutdir)

    for trajdir in traj_list:
        trajfulldir = join(env_root_dir, trajdir)
        save_vid_with_statistics((trajfulldir, vidoutdir))

def save_preview_video_mp(env_root_dir, traj_list, num_proc, vid_out_dir = 'video'):
    '''
    Input: Trajectory folder is organized in <env_root_dir>/data_folder[k]/P00X
    Output: a video for each trajectory: <vid_out_dir>/<datafolder>_P00X.mp4
    '''
    vidoutdir = join(env_root_dir, vid_out_dir)
    if not isdir(vidoutdir):
        mkdir(vidoutdir)
    params = []
    for trajdir in traj_list:
        trajfulldir = join(env_root_dir, trajdir)
        params.append([trajfulldir, vidoutdir])
    
    with Pool(num_proc) as pool:
        pool.map(save_vid_with_statistics, params) 

def write_datafile(datafile, trajinds, trajdir):
    trajlen = len(trajinds)
    datafile.write(trajdir)
    datafile.write(' ')
    datafile.write(str(trajlen))
    datafile.write('\n')
    for indstr in trajinds:
        datafile.write(indstr)
        datafile.write('\n')

def read_datafile(datafile):
    '''
    parse the datafile
    '''
    with open(datafile,'r') as f:
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
                print("Datafile Error: {}, line {}...".format(datafile, ind))
                raise Exception("Datafile Error: {}, line {}...".format(datafile, ind))
            line = lines[ind].strip()
            frames.append(line)
            ind += 1
        framelist.append(frames)
    totalframenum = sum(trajlenlist)
    print('    {}: Read {} trajectories, including {} frames'.format(datafile, len(trajlist), totalframenum))
    return trajlist, trajlenlist, framelist, totalframenum

def generate_datafile(env_root_dir, traj_list, ana_out_dir):
    '''
    env_root_dir: the root folder for one environment
    traj_list: a list of trajectory folders [DATAFOLDER/P00X]
    In each trajectory folder, there are multiple depth folders 'depth_lcam_front', 'depth_lcam_back', 'depth_lcam_right'... 
    Output index file: <env_root_dir>/<ana_out_dir>/left_file_index_all.txt 
            - this file is consistant with the dataloader's input file
            - content: 
                * firstline: <env_root_dir>/<datafolder>/<trajfolder> <num_of_frames>
                * each line correspond to a file index: 000xxx
                * each line does not contain image suffix
    Assume a data folder for rgb image exists
    '''
    # check and create output folders
    _, envstr = split(env_root_dir)
    indexfile = join(ana_out_dir, 'data_' + envstr + '.txt')
    outdatafile = open(indexfile, 'w')
    for trajdir in traj_list:
        trajfulldir = join(env_root_dir, trajdir)
        tempstrs, trajstr = split(trajdir)
        _, datastr = split(tempstrs)
        modfolder_dict = enumerate_modalities(trajfulldir)
        rgbfolderlist = modfolder_dict['Scene'] # hard coded, need to change for new airsim version
        framestrlist = enumerate_frames(join(trajfulldir, rgbfolderlist[0]))

        print('    Generate datafile for trajectory {}'.format(trajdir))
        indexfile_traj = join(ana_out_dir, 'data_' + envstr + '_' + datastr+'_'+trajstr+'.txt')
        with open(indexfile_traj, 'w') as f:
            write_datafile(f, framestrlist, join(envstr, trajdir))
        write_datafile(outdatafile, framestrlist, join(envstr, trajdir))
    outdatafile.close()

def analyze_depth_data(env_root_dir, traj_list, ana_out_dir, logf, info_from_file=False, num_proc=1):
    '''
    env_root_dir: the root folder for one environment
    traj_list: a list of trajectory folders [DATAFOLDER/P00X]
    In each trajectory folder, there are multiple depth folders 'depth_lcam_front', 'depth_lcam_back', 'depth_lcam_right'... 

    Output: 1. depth info: <env_root_dir>/<ana_out_dir>/disp_mean(max,min,std).npy
                - the numpy file is N x K, N is the number of frames in the whole environment, K is the number of cameras
            2. depth histogram: <env_root_dir>/<ana_out_dir>/disp_hist.jpg
            3. also save individule file for each trajectory
                - the numpy file is N x K, N is the number of frames, K is the number of cameras
                - the order is the same as the camera_list order defined in the collection_config.yaml
    '''
    
    # Convert the num_proc to proper value.
    if num_proc == 1:
        num_proc = None
    
    # Figure out the log filename if run in parallel.
    log_fn_parts = get_filename_parts(logf.filename)
    mp_log_fn = join( log_fn_parts[0], 'error_depth_mp.log' )
    
    # check and create output folders
    _, envstr = split(env_root_dir)

    # depth statistics
    dispmax = []
    dispmin = []
    dispmean = []
    dispstd = []
    
    # leftindexlist = []
    dispmaxfile = join(ana_out_dir, 'disp_max.npy')
    dispminfile = join(ana_out_dir, 'disp_min.npy')
    dispmeanfile = join(ana_out_dir, 'disp_mean.npy')
    dispstdfile = join(ana_out_dir, 'disp_std.npy')
    disphistfig = join(ana_out_dir, 'disp_hist.png')
    # indexfile = join(ana_out_dir, 'data_' + envstr + '.txt')
    # outdatafile = open(indexfile, 'w')

    if info_from_file:
        dispmax = np.load(dispmaxfile)
        dispmean = np.load(dispmeanfile)
        dispmin = np.load(dispminfile)
        dispstd = np.load(dispstdfile)
        print ('    depth info file loaded from file: {}, {}, {}, {}'.format(dispmax.shape, dispmin.shape, dispmean.shape, dispstd.shape))

    else: # calculate the statistics from depth image
        for trajdir in traj_list:
            trajfulldir = join(env_root_dir, trajdir)
            print('    Move into trajectory {}'.format(trajdir))
            tempstrs, trajstr = split(trajdir)
            _, datastr = split(tempstrs)

            traj_dmean, traj_dmax, traj_dmin, traj_dstd, fileindlist = verify_traj_depth(trajfulldir, logf, num_proc=num_proc, log_file=mp_log_fn)

            # also save statistics for each trajectory
            dispmaxfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_disp_max.npy')
            dispminfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_disp_min.npy')
            dispmeanfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_disp_mean.npy')
            dispstdfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_disp_std.npy')
            disphistfig_traj = join(ana_out_dir, datastr+'_'+trajstr+'_disp_hist.png')

            np.save(dispmeanfile_traj, np.array(traj_dmean))
            np.save(dispstdfile_traj, np.array(traj_dstd))
            np.save(dispmaxfile_traj, np.array(traj_dmax))
            np.save(dispminfile_traj, np.array(traj_dmin))
            # write_datafile(outdatafile, fileindlist, join(envstr, trajdir))
            plot_depth_info(traj_dmax, traj_dmin, traj_dmean, traj_dstd, disphistfig_traj, binnum=100)

            dispmax.append(traj_dmax)
            dispmin.append(traj_dmin)
            dispmean.append(traj_dmean)
            dispstd.append(traj_dstd)

        # generate depth statistic files
        dispmean = np.array(dispmean, dtype=object)
        dispmax = np.array(dispmax, dtype=object)
        dispmin = np.array(dispmin, dtype=object)
        dispstd = np.array(dispstd, dtype=object)
        np.save(dispmeanfile, dispmean)
        np.save(dispstdfile, dispstd)
        np.save(dispmaxfile, dispmax)
        np.save(dispminfile, dispmin)

        print ('    saved depth info file to {}: {}, {}, {}, {}'.format(ana_out_dir, 
                        split(dispmeanfile)[1], split(dispmaxfile)[1], split(dispminfile)[1], split(dispstdfile)[1]))
        # outdatafile.close()

    plot_depth_info(dispmax, dispmin, dispmean, dispstd, disphistfig)

def analyze_rgb_data(env_root_dir, traj_list, ana_out_dir, logf, info_from_file=False, num_proc=1):
    '''
    env_root_dir: the root folder for one environment
    traj_list: a list of trajectory folders [DATAFOLDER/P00X]
    In each trajectory folder, there are multiple image folders 'image_lcam_front', 'image_lcam_back', 'image_lcam_right'... 

    Output: 1. RGB info: <env_root_dir>/<ana_out_dir>/rgb_mean(max,min,std).npy
                - the numpy file is N x K, N is the number of frames in the whole environment, K is the number of cameras
            2. RGB histogram: <env_root_dir>/<ana_out_dir>/rgb_curve.jpg
            3. also save individule file for each trajectory
                - the numpy file is N x K, N is the number of frames, K is the number of cameras
                - the order is the same as the camera_list order defined in the collection_config.yaml

    Output: 1. rgb info: <env_root_dir>/<ana_out_dir>/rgb_mean(std).npy
            2. also save individule file for each trajectory
    '''
    
    # Convert the num_proc to proper value.
    if num_proc == 1:
        num_proc = None
    
    # Figure out the log filename if run in parallel.
    log_fn_parts = get_filename_parts(logf.filename)
    mp_log_fn = join( log_fn_parts[0], 'error_rgb_mp.log' )
    
    # RGB statistics
    rgbmax = []
    rgbmin = []
    rgbmean = []
    rgbstd = []
    
    # leftindexlist = []
    rgbmaxfile = join(ana_out_dir, 'rgb_max.npy')
    rgbminfile = join(ana_out_dir, 'rgb_min.npy')
    rgbmeanfile = join(ana_out_dir, 'rgb_mean.npy')
    rgbstdfile = join(ana_out_dir, 'rgb_std.npy')

    if info_from_file:
        rgbmax = np.load(rgbmaxfile)
        rgbmean = np.load(rgbmeanfile)
        rgbmin = np.load(rgbminfile)
        rgbstd = np.load(rgbstdfile)
        print ('    RGB info file loaded from file: {}, {}, {}, {}'.format(rgbmax.shape, rgbmin.shape, rgbmean.shape, rgbstd.shape))

    else: # calculate the statistics from RGB image
        for trajdir in traj_list:
            trajfulldir = join(env_root_dir, trajdir)
            print('    Move into trajectory {}'.format(trajdir))
            tempstrs, trajstr = split(trajdir)
            _, datastr = split(tempstrs)

            traj_mean, traj_max, traj_min, traj_std, fileindlist = verify_traj_rgb(trajfulldir, logf, num_proc=num_proc, log_file=mp_log_fn)

            # also save statistics for each trajectory
            rgbmaxfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_rgb_max.npy')
            rgbminfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_rgb_min.npy')
            rgbmeanfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_rgb_mean.npy')
            rgbstdfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_rgb_std.npy')
            rgbhistfig_traj = join(ana_out_dir, datastr+'_'+trajstr+'_rgb_mean_std.png')

            np.save(rgbmeanfile_traj, np.array(traj_mean))
            np.save(rgbstdfile_traj, np.array(traj_std))
            np.save(rgbmaxfile_traj, np.array(traj_max))
            np.save(rgbminfile_traj, np.array(traj_min))
            plot_rgb_info(traj_mean, traj_std, rgbhistfig_traj)

            rgbmax.append(traj_max)
            rgbmin.append(traj_min)
            rgbmean.append(traj_mean)
            rgbstd.append(traj_std)

        # generate RGB statistic files
        rgbmean = np.array(rgbmean, dtype=object)
        rgbmax = np.array(rgbmax, dtype=object)
        rgbmin = np.array(rgbmin, dtype=object)
        rgbstd = np.array(rgbstd, dtype=object)
        np.save(rgbmeanfile, rgbmean)
        np.save(rgbstdfile, rgbstd)
        np.save(rgbmaxfile, rgbmax)
        np.save(rgbminfile, rgbmin)

        print ('    saved RGB info file to {}: {}, {}, {}, {}'.format(ana_out_dir, 
                        split(rgbmeanfile)[1], split(rgbmaxfile)[1], split(rgbminfile)[1], split(rgbstdfile)[1]))


def analyze_seg_data(framewriter, env_root_dir, traj_list, label_file, ana_out_dir, logf, num_proc=1):
    '''
    env_root_dir: the root folder for one environment
    traj_list: a list of trajectory folders [DATAFOLDER/P00X]
    In each trajectory folder, there are multiple image folders 'image_lcam_front', 'image_lcam_back', 'image_lcam_right'... 

    Output: 1. Seg info: <env_root_dir>/<ana_out_dir>/seg.npy
                - the numpy file is dictionary, 
                    key is the class label, 
                    value is an N x K array, indicating the number of pixels in one image
                - also generate a global seg file for the whole environment
                    key is the class label
                    value is the total number of pixels in all images
                - <env_root_dir>/<ana_out_dir>/datastr_trajstr_segdata.npy: 
                    N x 12 x 256 bool matrix recording the seg data in each frame, where N is the number of frames in one traj
                    this matrix store what seg label is appared in each frame
    ''' 
    
    # Convert the num_proc to proper value.
    if num_proc == 1:
        num_proc = None
    
    # Figure out the log filename if run in parallel.
    log_fn_parts = get_filename_parts(logf.filename)
    mp_log_fn = join( log_fn_parts[0], 'error_seg_mp.log' )
    
    assert isfile(label_file), 'Cannot find seg label file {}'.format(label_file)   
    segfile = join(ana_out_dir, 'seg.yaml')
    segerrorfilename = join(ana_out_dir, 'seg_error.txt')
    segerrorfile = open(segerrorfilename, 'w')

    badframedir = join(ana_out_dir, 'bad_frames')
    if not isdir(badframedir):
        mkdir(badframedir)
    badframedir_seg = join(badframedir, 'bad_segs')
    if not isdir(badframedir_seg):
        mkdir(badframedir_seg)


    allpixelnums = {}
    # calculate the statistics from seg image
    for trajdir in traj_list:
        trajfulldir = join(env_root_dir, trajdir)
        print('    Move into trajectory {}'.format(trajdir))
        tempstrs, trajstr = split(trajdir)
        _, datastr = split(tempstrs)
        
        pixelnums, segdata, errordict = verify_traj_seg(trajfulldir, label_file, logf, num_proc=num_proc, log_file=mp_log_fn)

        # also save statistics for each trajectory
        segfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_seg.npy')
        segdatafile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_segdata.npy')
        segerrorfile_traj = join(ana_out_dir, datastr+'_'+trajstr+'_seg_error.txt')

        np.save(segfile_traj, pixelnums)
        np.save(segdatafile_traj, segdata)

        for seglabel, pixelnum in pixelnums.items():
            if seglabel not in allpixelnums:
                allpixelnums[seglabel] = int(np.sum(pixelnum.astype(np.int64)))
            else:
                allpixelnums[seglabel] += int(np.sum(pixelnum.astype(np.int64)))

        # # write debugging image to the folder
        # SegSaveLimit = 10
        # for color, framelist in errordict.items():
        #     segerrorfile.write("Color {} in traj {} frames {} ...\n".format(color, trajdir, framelist[:10]))
        #     for k, framestr in enumerate(framelist):
        #         framewriter.write_seg_with_id(badframedir_seg, trajdir, framestr, color)
        #         if k>= SegSaveLimit:
        #             break
        
        # write the bad frames to a txt file
        trajerrorfile = SegErrorFileWriter(segerrorfile_traj)
        trajerrorfile.dump(errordict)
        trajerrorfile.close()
        print('  ---')

    # generate seg statistic files
    with open(segfile, 'w') as outfile:
        yaml.dump(allpixelnums, outfile, default_flow_style=False)
    # np.savetxt(segfile, allpixelnums)
    print ('    saved Seg info file to {}: {}'.format(ana_out_dir, split(segfile)[1]))

    segerrorfile.close()

class FrameWriter(object):
    '''
    write bad frames and the statistics info to folders for debugging
    '''
    def __init__(self) -> None:
        
        self.rgbfolderlist_left = [
            'image_lcam_front',
            'image_lcam_back',
            'image_lcam_right',
            'image_lcam_left',
        ]
        self.rgbfolderlist_right = [
            'image_rcam_front',
            'image_rcam_back',
            'image_rcam_right',
            'image_rcam_left',
        ]
        self.depthfolderlist_left = [
            'depth_lcam_front',
            'depth_lcam_back',
            'depth_lcam_right',
            'depth_lcam_left',
        ]
        self.depthfolderlist_right = [
            'depth_rcam_front',
            'depth_rcam_back',
            'depth_rcam_right',
            'depth_rcam_left',
        ]
        self.segfolderlist = [
            'seg_lcam_front',
            'seg_lcam_back',
            'seg_lcam_right',
            'seg_lcam_left',
            'seg_lcam_top',
            'seg_lcam_bottom',
            'seg_rcam_front',
            'seg_rcam_back',
            'seg_rcam_right',
            'seg_rcam_left',
            'seg_rcam_top',
            'seg_rcam_bottom',
        ]
        self.imgreader = ImageReader()
        self.imgvisualizer = DataVisualizer()
        self.color_to_ind = None
        self.ind_to_color = None
        self.envdir = None

    def init_env(self, envdir):
        '''
        envdir: the full path of the environment folder
        '''
        self.envdir = envdir
        labelfile = join(envdir, 'seg_label.json')
        self.color_to_ind, segname_to_ind, segind_to_name = get_seg_color_label_value(labelfile)
        self.ind_to_color = {self.color_to_ind[k]: k for k in self.color_to_ind}
      
    def write_rgb(self, outputfolder, trajfolder, framestr, rgb_mean, rgb_std):
        '''
        trajfolder: relative path wrt the env: Data_easy/P000
        rgb_statistics: rgb mean [12], rgb std [12]
        output a vis image:  rgb 0-3
        '''
        if self.envdir is None:
            print("Error: init_env has not been called!")
            return
        trajdir = join(self.envdir, trajfolder)
        visimgs = vis_frame(self.imgreader, self.imgvisualizer, 
                            trajdir, framestr, scale=0.5, 
                            depthfolderlist=None, rgbfolderlist=self.rgbfolderlist_left)
        text = "RGB mean {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ".format(rgb_mean[0], rgb_mean[1], rgb_mean[2], rgb_mean[3], rgb_mean[4], rgb_mean[5] )
        text = text + "| {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(rgb_mean[6], rgb_mean[7], rgb_mean[8], rgb_mean[9], rgb_mean[10], rgb_mean[11] )
        visimgs = self.imgvisualizer.add_text(visimgs, text) 
        text = "RGB std  {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ".format(rgb_std[0], rgb_std[1], rgb_std[2], rgb_std[3], rgb_std[4], rgb_std[5] )
        text = text + "| {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(rgb_std[6], rgb_std[7], rgb_std[8], rgb_std[9], rgb_std[10], rgb_std[11] )
        visimgs = self.imgvisualizer.add_text(visimgs, text, offset_height=30) 

        datastr, trajstr = split(trajfolder)
        outputfile = join(outputfolder, 'bad_rgb_'+ datastr+ '_' + trajstr + '_' +framestr+'.png')
        cv2.imwrite(outputfile, visimgs)

    def write_depth(self, outputfolder, trajfolder, framestr, disp_max, disp_mean):
        '''
        trajfolder: relative path wrt the env: Data_easy/P000
        disp_max: 12 vector
        '''
        if self.envdir is None:
            print("Error: init_env has not been called!")
            return

        # decide between left or right frames
        if disp_max[:6].max() > disp_max[6:].max():
            depthfolderlist = self.depthfolderlist_left
            rgbfolderlist = self.rgbfolderlist_left
        else:
            depthfolderlist = self.depthfolderlist_right
            rgbfolderlist = self.rgbfolderlist_right

        trajdir = join(self.envdir, trajfolder)
        visimgs = vis_frame(self.imgreader, self.imgvisualizer, 
                            trajdir, framestr, scale=0.5, 
                            depthfolderlist=depthfolderlist, rgbfolderlist=rgbfolderlist)
        text = "Disp max  {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ".format(disp_max[0], disp_max[1], disp_max[2], disp_max[3], disp_max[4], disp_max[5] )
        text = text + "| {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(disp_max[6], disp_max[7], disp_max[8], disp_max[9], disp_max[10], disp_max[11] )
        visimgs = self.imgvisualizer.add_text(visimgs, text) 
        text = "Disp mean {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} ".format(disp_mean[0], disp_mean[1], disp_mean[2], disp_mean[3], disp_mean[4], disp_mean[5] )
        text = text + "| {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(disp_mean[6], disp_mean[7], disp_mean[8], disp_mean[9], disp_mean[10], disp_mean[11] )
        visimgs = self.imgvisualizer.add_text(visimgs, text, offset_height=30) 

        datastr, trajstr = split(trajfolder)
        outputfile = join(outputfolder, 'bad_disp_'+ datastr+ '_' + trajstr + '_' +framestr+'.png')
        cv2.imwrite(outputfile, visimgs)


    def write_seg_with_id(self, outputfolder, trajfolder, framestr, seg_color):
        '''
        seg_id: the problematic id that needs to be visualized
        seg_data: 256 vector indicating which label exists in each image
        '''
        if self.envdir is None:
            print("Error: init_env has not been called!")
            return

        flag = False
        for segfolder in self.segfolderlist:
            segfilesurfix = segfolder.split('seg_')[-1] + '_seg.png'
            segfile = join(self.envdir, trajfolder, segfolder, framestr + '_' + segfilesurfix)

            # if seg_color not in self.color_to_ind: 

            # if seg_data[seg_id] == 0: # the seg_id does not appared in the frame
            #     print("Error: the seg_id {} is not in the frame {}".format(seg_id, segfile))
            #     return
            # if seg_id not in self.ind_to_color:
            #     print("Error: the seg_id {} in not in the seg_rgbs list".format(seg_id))
            #     return

            segimg = cv2.imread(segfile, cv2.IMREAD_UNCHANGED)
            # seg_color = self.ind_to_color[seg_id]
            seg_mask = segimg == seg_color
            if seg_mask.sum() == 0:
                continue # this image does not contain the seg_color

            vis = np.zeros_like(segimg, dtype=np.uint8)
            vis[seg_mask] = 255
            vis = np.repeat(vis[...,np.newaxis], 3, axis=2) # h x w x 3
            vis = cv2.resize(vis, (0,0), fx=0.5, fy=0.5,  interpolation=cv2.INTER_NEAREST)

            rgbimgfile = join(segfile.replace('seg_', 'image_').replace('_seg.png', '.png')) # hard code
            rgbimg = cv2.imread(rgbimgfile)

            rgbimg2 = cv2.resize(rgbimg, (vis.shape[0], vis.shape[1]))
            segmask = vis == 255
            segoverlay = rgbimg2 * 0.3
            segoverlay[segmask] = rgbimg2[segmask] 

            visimgs = np.concatenate((rgbimg, np.concatenate((vis, segoverlay), axis=0)), axis=1)

            seg_id = self.color_to_ind[seg_color] if seg_color in self.color_to_ind else -1 
            text = "Seg id {} color {}".format(seg_id, seg_color)
            visimgs = self.imgvisualizer.add_text(visimgs, text)
            text = "File {}".format(segfile)
            visimgs = self.imgvisualizer.add_text(visimgs, text, offset_height=30)
            datastr, trajstr = split(trajfolder)
            outputfile = join(outputfolder, 'bad_seg_'+ datastr+ '_' + trajstr + '_' + segfolder + '_' + framestr+ '_color_' + str(seg_color)+ '_id_' + str(seg_id) +'.png')
            cv2.imwrite(outputfile, visimgs)

            flag = True
            break

        if not flag:
            print("Error: cannot file color {} in frame {}/{}".format(seg_color, trajfolder, framestr))

def seg_filter(segerrorfile, framewriter, badframedir_seg, trajfolder, framestrs, SegSaveLimit=5):
    '''
    Input: 
        anafolder: 'analyze'
        text file recording the seg error color and frames 
        list of image indexes
    Output: 
        - save debugging file into the folder
    '''
    # read statistic files from the analyze folder, and put it on the image
    trajsegerrorfile = SegErrorFileReader(segerrorfile)
    errordict = trajsegerrorfile.load()
    
    framenum = len(framestrs)
    # write debugging image to the folder
    bagsegcount = 0
    outputfilecount = 0
    for color, framelist in errordict.items():
        # segerrorfile.write("Color {} in traj {} frames {} ...\n".format(color, trajfolder, framelist[:10]))
        bagsegcount += len(framelist)
        for k, framestr in enumerate(framelist):
            framewriter.write_seg_with_id(badframedir_seg, trajfolder, framestr, color)
            outputfilecount += 1
            if k>= SegSaveLimit-1:
                break

    print ('    Write {}/{}/{} bad seg files to {}'.format(outputfilecount, bagsegcount, framenum, trajfolder))
    print ('  ----')

def seg_filter_all_trajs(framewriter, env_root_dir, trajlist, ana_out_dir = 'analyze'):
    '''
    Iterate the trajectory
    Read the xxx_seg_error.txt
    Output visualization to the folder
    '''
    # check and create output folders
    anaoutdir = join(env_root_dir, ana_out_dir)
    _, envname = split(env_root_dir)

    assert isdir(anaoutdir), 'Missing analyze folder, please run data_validation first!'

    # make folders for storing the visualization of bad frames 
    badframedir = join(anaoutdir, 'bad_frames')
    if not isdir(badframedir):
        mkdir(badframedir)
    badframedir_seg = join(badframedir, 'bad_segs')
    if not isdir(badframedir_seg):
        mkdir(badframedir_seg)

    for trajdir in trajlist:
        print('    Filter trajectory by seg {}'.format(trajdir))
        tempstrs, trajstr = split(trajdir)
        _, datastr = split(tempstrs)
        segerrorfile_traj = join(anaoutdir, datastr+'_'+trajstr+'_seg_error.txt')

        fileprefix = datastr+'_'+trajstr
        datafile = join(anaoutdir, 'data_' + envname + '_' + fileprefix+'.txt')
        _, _, framelist, _ = read_datafile(datafile) # read the framestrs

        seg_filter(segerrorfile_traj, framewriter, badframedir_seg, trajdir, framelist[0])

def depth_rgb_filter(dispmaxfile, dispmeanfile, rgbmeanfile, rgbstdfile,
                        framewriter, badframedir_rgb, badframedir_depth, trajfolder, framestrs, 
                        max_disp_max = 500, max_disp_mean = 250, # the objects can not be too close
                        min_disp_max = 4, min_disp_mean = 0.4, # the objects can not be all far away (e.g. looking into sky)
                        min_rgb_mean = 5, min_rgb_std = 5, # the scene can not be too dark
                        max_rgb_mean = 240, min_rgb_std2 = 5, # the scene can not be too bright (e.g. facing a white wall) 
                        max_disp_max2 = 800,
                        ):
    '''
    Input: 
        anafolder: 'analyze'
        npy files of depth info 
        list of image indexes
    Output: 
        - bad frames: a few datafile after filterring the bad frames
        - out_traj_error_file: bad frames and the error code

    Depth filterring condition
        - Nothing too close (maxmax=500 -> min_dist=80/500=0.16m)
        - No big thing too close (maxmean=250 -> mean_dist=80/250=0.32m)
        - No all background (minmax=4 -> max_dist=80/4=20m) 
        - Large enough close object (minmean=0.4 -> mean_dist=80/0.4=200m)
    RGB filterring condition
        - Not too dark
        - Not too bright
    Error code:
        - 0: small object very close
        - 1: large object too close
        - 2: scene too far
        - 3: rgb too dark
        - 4: rgb too bright
    '''
    # read statistic files from the analyze folder, and put it on the image
    dispmax = np.load(dispmaxfile)
    dispmean = np.load(dispmeanfile)

    rgbmean = np.load(rgbmeanfile)
    rgbstd = np.load(rgbstdfile)

    framenum = len(dispmax)
    print ('    input file loaded (maxdisp, meandisp, rgbmean, rgbstd): {}, {}, {}, {}'.format(dispmax.shape, 
                    dispmean.shape, rgbmean.shape, rgbstd.shape))

    count = 0
    goodframes = []
    goodframes_subseq = []
    badframes = {}
    badrgbcount, baddispcount, badsegcount = 0, 0, 0
    for k in range(framenum):
        errorlist = []

        if dispmax[k].max() > max_disp_max2 or dispmax[k].max() > max_disp_max2: 
            errorlist.append(0)

        if dispmax[k].mean() > max_disp_max or dispmax[k].mean() > max_disp_max or \
            dispmean[k].mean() > max_disp_mean or dispmean[k].mean() > max_disp_mean: 
            errorlist.append(1)

        if dispmax[k].mean() < min_disp_max or dispmax[k].mean() < min_disp_max or \
            dispmean[k].mean() < min_disp_mean or dispmean[k].mean() < min_disp_mean: 
            errorlist.append(2)

        if len(errorlist) > 0:
            framewriter.write_depth(badframedir_depth, trajfolder, framestrs[k], dispmax[k], dispmean[k])
            baddispcount += 1

        if rgbmean[k].mean() < min_rgb_mean and rgbstd[k].mean() < min_rgb_std:
            errorlist.append(3)

        if rgbmean[k].mean() > max_rgb_mean and rgbstd[k].mean() < min_rgb_std:
            errorlist.append(4)

        if 3 in errorlist or 4 in errorlist:
            framewriter.write_rgb(badframedir_rgb, trajfolder, framestrs[k], rgbmean[k], rgbstd[k])
            badrgbcount += 1

        if len(errorlist)==0:
            goodframes_subseq.append(framestrs[k])
            count += 1
        else: 
            badframes[framestrs[k]] = errorlist
            if len(goodframes_subseq) > 0:
                goodframes.append(goodframes_subseq)
                goodframes_subseq = []

    if len(goodframes_subseq) > 0:
        goodframes.append(goodframes_subseq)
        goodframes_subseq = []

    print ('    Filtering: {}/{}, total bad rgb {}, bad depth {}'.format(count, framenum, badrgbcount, baddispcount))
    print ('  ----')
    return goodframes, badframes

def write_bad_datafile(datafile, trajinds, trajdir):
    '''
    trajdir framenum
    frameind0 errorcode
    frameind1 errorcode
    ...
    '''
    trajlen = len(trajinds)
    datafile.write(trajdir)
    datafile.write(' ')
    datafile.write(str(trajlen))
    datafile.write('\n')
    for indstr in trajinds:
        datafile.write(indstr)
        datafile.write(' ')
        for errornum in trajinds[indstr]:
            datafile.write(str(errornum))
            datafile.write(' ')
        datafile.write('\n')

def depth_rgb_filter_all_trajs(framewriter, env_root_dir, trajlist, ana_out_dir = 'analyze'):
    '''
    Iterate the trajectory and call depth_rgb_filter
    '''
    # check and create output folders
    anaoutdir = join(env_root_dir, ana_out_dir)
    _, envname = split(env_root_dir)

    assert isdir(anaoutdir), 'Missing analyze folder, please run data_validation first!'

    out_env_file = open(join(anaoutdir, 'data_' + envname+'_good_frames.txt'), 'w')
    out_env_bad_file = open(join(anaoutdir, 'data_' + envname+'_bad_frames.txt'), 'w')
    out_env_bad_file.write('# Error code: 0: small object very close; 1: large object too close; 2: scene too far; 3: rgb too dark; 4: rgb too bright\n')

    # make folders for storing the visualization of bad frames 
    badframedir = join(anaoutdir, 'bad_frames')
    if not isdir(badframedir):
        mkdir(badframedir)
    badframedir_rgb = join(badframedir, 'bad_rgbs')
    if not isdir(badframedir_rgb):
        mkdir(badframedir_rgb)
    badframedir_depth = join(badframedir, 'bad_depths')
    if not isdir(badframedir_depth):
        mkdir(badframedir_depth)

    for trajdir in trajlist:
        print('    Filter trajectory {}'.format(trajdir))
        tempstrs, trajstr = split(trajdir)
        _, datastr = split(tempstrs)

        fileprefix = datastr+'_'+trajstr
        dispmaxfile = join(anaoutdir, fileprefix+'_disp_max.npy')
        dispmeanfile = join(anaoutdir, fileprefix+'_disp_mean.npy')
        rgbmeanfile = join(anaoutdir, fileprefix+'_rgb_mean.npy')
        rgbstdfile = join(anaoutdir, fileprefix+'_rgb_std.npy')
        datafile = join(anaoutdir, 'data_' + envname + '_' + fileprefix+'.txt')

        _, _, framelist, _ = read_datafile(datafile) # read the framestrs
        goodframes, badframes = depth_rgb_filter(dispmaxfile, dispmeanfile, rgbmeanfile, rgbstdfile, 
                                                framewriter, badframedir_rgb, badframedir_depth, join(datastr, trajstr), framelist[0])

        for goodframes_subseq in goodframes:
            write_datafile(out_env_file, goodframes_subseq, trajdir)
        if len(badframes)>0:
            write_bad_datafile(out_env_bad_file, badframes, trajdir )

    out_env_file.close()
    out_env_bad_file.close()

def update_all_trajs_after_remove(env_dir, trajlist, ana_out_dir = 'analyze'):
    '''
    1. read per-traj depth statistics files combine them: disp_max, disp_min, disp_mean, disp_std, disp_hist
    2. read per-traj rgb statistics files combine them: rgb_max, rgb_mean, rgb_min, rgb_std
    3. read per-traj seg statistics files combine them: seg.yaml
    '''
    anaoutdir = join(env_dir, ana_out_dir)
    _, envname = split(env_dir)

    assert isdir(anaoutdir), 'Missing analyze folder, please run data_validation first!'

    # Disp statistics
    dispmax = []
    dispmin = []
    dispmean = []
    dispstd = []
    dispmaxfile = join(anaoutdir, 'disp_max.npy')
    dispminfile = join(anaoutdir, 'disp_min.npy')
    dispmeanfile = join(anaoutdir, 'disp_mean.npy')
    dispstdfile = join(anaoutdir, 'disp_std.npy')
    disphistfig = join(anaoutdir, 'disp_hist.png')

    # RGB statistics
    rgbmax = []
    rgbmin = []
    rgbmean = []
    rgbstd = []
    
    rgbmaxfile = join(anaoutdir, 'rgb_max.npy')
    rgbminfile = join(anaoutdir, 'rgb_min.npy')
    rgbmeanfile = join(anaoutdir, 'rgb_mean.npy')
    rgbstdfile = join(anaoutdir, 'rgb_std.npy')

    # Seg pixel file
    allpixelnums = {}
    segfile = join(anaoutdir, 'seg.yaml')

    for trajdir in trajlist:
        print('    Update trajectory {}'.format(trajdir))
        tempstrs, trajstr = split(trajdir)
        _, datastr = split(tempstrs)
        fileprefix = datastr+'_'+trajstr

        # read per-traj disp data
        dispmaxfile_traj = join(anaoutdir, fileprefix+'_disp_max.npy')
        dispminfile_traj = join(anaoutdir, fileprefix+'_disp_min.npy')
        dispmeanfile_traj = join(anaoutdir, fileprefix+'_disp_mean.npy')
        dispstdfile_traj = join(anaoutdir, fileprefix+'_disp_std.npy')

        traj_dmax = np.load(dispmaxfile_traj)
        traj_dmin = np.load(dispminfile_traj)
        traj_dmean = np.load(dispmeanfile_traj)
        traj_dstd = np.load(dispstdfile_traj)

        dispmax.append(traj_dmax)
        dispmin.append(traj_dmin)
        dispmean.append(traj_dmean)
        dispstd.append(traj_dstd)

        # read per-traj rgb data
        rgbmaxfile_traj = join(anaoutdir, fileprefix+'_rgb_max.npy')
        rgbminfile_traj = join(anaoutdir, fileprefix+'_rgb_min.npy')
        rgbmeanfile_traj = join(anaoutdir, fileprefix+'_rgb_mean.npy')
        rgbstdfile_traj = join(anaoutdir, fileprefix+'_rgb_std.npy')

        traj_max_rgb = np.load(rgbmaxfile_traj)
        traj_min_rgb = np.load(rgbminfile_traj)
        traj_mean_rgb = np.load(rgbmeanfile_traj)
        traj_std_rgb = np.load(rgbstdfile_traj)

        rgbmax.append(traj_max_rgb)
        rgbmin.append(traj_min_rgb)
        rgbmean.append(traj_mean_rgb)
        rgbstd.append(traj_std_rgb)

        # read per-traj seg data
        segfile_traj = join(anaoutdir, fileprefix+'_seg.npy')
        pixelnums = np.load(segfile_traj, allow_pickle=True)
        pixelnums = pixelnums.item()

        for seglabel, pixelnum in pixelnums.items():
            if seglabel not in allpixelnums:
                allpixelnums[seglabel] = int(np.sum(pixelnum.astype(np.int64)))
            else:
                allpixelnums[seglabel] += int(np.sum(pixelnum.astype(np.int64)))

    dispmean = np.array(dispmean, dtype=object)
    dispmax = np.array(dispmax, dtype=object)
    dispmin = np.array(dispmin, dtype=object)
    dispstd = np.array(dispstd, dtype=object)
    np.save(dispmeanfile, dispmean)
    np.save(dispstdfile, dispstd)
    np.save(dispmaxfile, dispmax)
    np.save(dispminfile, dispmin)

    plot_depth_info(dispmax, dispmin, dispmean, dispstd, disphistfig)

    rgbmean = np.array(rgbmean, dtype=object)
    rgbmax = np.array(rgbmax, dtype=object)
    rgbmin = np.array(rgbmin, dtype=object)
    rgbstd = np.array(rgbstd, dtype=object)
    np.save(rgbmeanfile, rgbmean)
    np.save(rgbstdfile, rgbstd)
    np.save(rgbmaxfile, rgbmax)
    np.save(rgbminfile, rgbmin)

    with open(segfile, 'w') as outfile:
        yaml.dump(allpixelnums, outfile, default_flow_style=False)

# python -m postprocessing.data_validation --data-root /home/amigo/tmp/test_root --data-folders Data_easy 
# --analyze-depth --analyze-rgb 
# --analyze-seg --seg-label-file FILEPATH
# --rgb-depth-filter --update-ana-files

if __name__ == '__main__':
    from settings import get_args
    from .data_enumeration import enumerate_trajs

    args = get_args()
    data_root_dir   = args.data_root
    data_folders    = args.data_folders.split(',')
    env_folders     = args.env_folders.split(',')
    traj_folders    = args.traj_folders.split(',')

    create_video    = args.create_video

    analyze_depth   = args.analyze_depth
    depth_from_file = args.depth_from_file
    analyze_rgb     = args.analyze_rgb
    analyze_seg     = args.analyze_seg
    label_file      = args.seg_label_file
    update_ana_files= args.update_ana_files

    rgb_depth_filter = args.rgb_depth_filter
    
    # Multiprocessing.
    num_proc = args.np if args.np > 1 else None

    trajdict = enumerate_trajs(data_root_dir, data_folders)
    if env_folders[0] == '': # env_folders not specified use all envs
        env_folders = list(trajdict.keys())
        
    # import ipdb;ipdb.set_trace()
    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        trajlist = trajdict[env_folder]
        print('Working on env {}'.format(env_dir))
        
        if  traj_folders[0]!='': # trajectories are specified by the user
            trajlist = [tt for tt in trajlist if split(tt)[-1] in traj_folders]

        anaoutdir = join(env_dir, 'analyze')
        if not isdir(anaoutdir):
            mkdir(anaoutdir)
        logf = FileLogger(join(anaoutdir,'error.log'))

        framewriter = FrameWriter()
        framewriter.init_env(env_dir)

        generate_datafile(env_dir, trajlist, ana_out_dir = anaoutdir)
        if analyze_depth:
            print ('  analyzing depth..')
            analyze_depth_data(env_dir, trajlist, ana_out_dir = anaoutdir, logf = logf, info_from_file=depth_from_file, num_proc=num_proc)
        if analyze_rgb:
            print('  analyzing rgb..')
            analyze_rgb_data(env_dir, trajlist, ana_out_dir = anaoutdir, logf = logf, info_from_file=depth_from_file, num_proc=num_proc)
        if analyze_seg:
            print('  analyzing seg..')
            env_label_file = join(env_dir, label_file)
            print('  opening seg label file: {}'.format(env_label_file))
            analyze_seg_data(framewriter, env_dir, trajlist, env_label_file, ana_out_dir = anaoutdir, logf = logf, num_proc=num_proc)

        if create_video:
            print ('  creating video..')
            save_preview_video_mp(env_dir, trajlist, num_proc=num_proc, vid_out_dir = 'video')

        if rgb_depth_filter:
            print ('  filtering depth and rgb..')
            depth_rgb_filter_all_trajs(framewriter, env_dir, trajlist, ana_out_dir = 'analyze')
            seg_filter_all_trajs(framewriter, env_dir, trajlist, ana_out_dir = 'analyze')

        if update_ana_files:
            print('  update the analysis files..')
            update_all_trajs_after_remove(env_dir, trajlist, ana_out_dir = 'analyze')

        logf.close()
        print('***** Finished env {} *****'.format(env_dir))
        print('')

    # # test filewriter
    # envdir = "E:\\TartanAir_v2\\SewerageExposure"

    # # make folders for storing the visualization of bad frames 
    # badframedir = join(envdir, 'bad_frames')
    # if not isdir(badframedir):
    #     mkdir(badframedir)
    # framewriter = FrameWriter()
    # framewriter.init_env(envdir)

    # framewriter.write_depth(badframedir, 'Data_easy\\P000', '000100', np.random.random(12), np.random.random(12))
    # framewriter.write_rgb(badframedir, 'Data_easy\\P000', '000100', np.random.random(12), np.random.random(12))
    # framewriter.write_seg_with_id(badframedir, 'Data_easy\\P000', '000100', 'seg_lcam_front', 15, np.ones(256, dtype=np.uint8))