
from datetime import datetime
import json
import numpy as np
from os.path import join, isfile, isdir
import time

from .ImageReader import ImageReader
from .data_enumeration import enumerate_modalities, enumerate_frames

from .process_pool import ( ReplicatedArgument, PoolWithLogger )

import os
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class MeasureTime(object):
    def __init__(self, name, indent=''):
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.indent = indent
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print("{}Time for {}: {:.2f} seconds.".format(self.indent, self.name, self.duration))

def find_folders_and_frames(trajdir, data_type, type_name_str):
    modfolder_dict = enumerate_modalities(trajdir)
    modfolderlist = modfolder_dict[data_type]
    for modfolder in modfolderlist:
        assert isdir(trajdir + '/' + modfolder), '{} folder {} not exists '.format(type_name_str, modfolder)

    # find the frames
    framestrlist = enumerate_frames(trajdir + '/' + modfolderlist[0])
    framenum = len(framestrlist)
    assert len(framestrlist) > 0, "Cannot find frames in {}".format(trajdir + '/' + modfolderlist[0])
    print("    Process {} folder {}, frame num {}, folder num {}".format(type_name_str, trajdir, framenum, len(modfolderlist)))
    
    return modfolderlist, framestrlist, framenum

def proc_verify_frame_init(logger_name, log_queue):
    global PROC_IMG_READER, PROC_LOGGER
    
    # The logger.
    PROC_LOGGER = PoolWithLogger.job_prepare_logger(logger_name, log_queue)
    
    # The image reader. 
    PROC_IMG_READER = ImageReader()

#================================#
#============ Depth =============#
#================================#

def proc_verify_frame_depth(trajdir, depthfolderlist, framestr):
    global PROC_IMG_READER, PROC_LOGGER
    
    dmax  = []
    dmin  = []
    dmean = []
    dstd  = []
    
    for _, depthfolder in enumerate(depthfolderlist):
        depthmodstr = depthfolder.split('depth_')[-1] # hard coded
        depthfile_surfix = depthmodstr + '_depth.png'
        depthfile = join(trajdir, depthfolder, framestr + '_' + depthfile_surfix)
        
        if isfile(depthfile):
            depthnp = PROC_IMG_READER.read_disparity(depthfile)
            dmax.append(  depthnp.max()  )
            dmin.append(  depthnp.min()  )
            dmean.append( depthnp.mean() )
            dstd.append(  depthnp.std()  )
        else:
            time_str = datetime.now().strftime("%m/%d/%y %H:%M:%S.%f")
            PROC_LOGGER.info(f'{time_str}: Depth file missing {depthfile}. ')
    
    return dict(
        dmax=dmax, 
        dmin=dmin, 
        dmean=dmean, 
        dstd=dstd, 
        framestr=framestr )

def parallel_verify_frame_dpeth(trajdir, startind, framenum, framestrlist, depthfolderlist, num_proc, log_file):
    traj_dmax   = []
    traj_dmin   = []
    traj_dmean  = []
    traj_dstd   = []
    fileindlist = []

    # Prepare the arguments.
    framestr_list = [ framestrlist[k] for k in range(startind, framenum) ]
    N = len(framestr_list)
    rep_trajdir = ReplicatedArgument(trajdir, N)
    rep_depthfolderlist = ReplicatedArgument(depthfolderlist, N)
    
    zipped_args = zip( rep_trajdir, rep_depthfolderlist, framestr_list )

    # Run in parallel.
    with PoolWithLogger( num_proc, proc_verify_frame_init, 'tartanair', log_file ) as pool:
        results = pool.map( proc_verify_frame_depth, zipped_args )
        
        for res in results:
            traj_dmax.append( res['dmax'] )
            traj_dmin.append( res['dmin'] )
            traj_dmean.append( res['dmean'] )
            traj_dstd.append( res['dstd'] )
            fileindlist.append( res['framestr'] )
            
    return traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist

def single_verify_frame_dpeth(logf, trajdir, startind, framenum, framestrlist, depthfolderlist):
    imgreader = ImageReader()
    
    traj_dmax   = []
    traj_dmin   = []
    traj_dmean  = []
    traj_dstd   = []
    fileindlist = []
    
    for k in range(startind, framenum):
        framestr = framestrlist[k]
        dmax  = []
        dmin  = []
        dmean = []
        dstd  = []
        for w, depthfolder in enumerate(depthfolderlist):
            depthmodstr = depthfolder.split('depth_')[-1] # hard coded
            depthfile_surfix = depthmodstr + '_depth.png'

            depthfile = join(trajdir, depthfolder, framestr + '_' + depthfile_surfix)
            if isfile(depthfile):
                depthnp = imgreader.read_disparity(depthfile)
            else:
                logf.logline('Depth file missing ' + depthfile)
                print ('Depth file missing', depthfile)
                continue

            dmax.append(  depthnp.max()  )
            dmin.append(  depthnp.min()  )
            dmean.append( depthnp.mean() )
            dstd.append(  depthnp.std()  )

        traj_dmax.append(dmax)
        traj_dmin.append(dmin)
        traj_dmean.append(dmean)
        traj_dstd.append(dstd)
        fileindlist.append(framestr)

        if k%100==0:
            print("    Read {} depth files...".format(k))
    
    print(f"    {framenum - startind} depth files processed. ")
            
    return traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist

def verify_traj_depth(trajdir, logf, startind=0, num_proc=None, log_file=None):
    '''
    return statistics on depth image
    dmean, dmax, dmin, dstd: N x K numpy array, where N is the number of frames of a single trajectory,
        K is the number of depth images in a single frame.
    frameinds: N strings
    num_proc: number of processes. Use None to disable multiprocessing.
    log_file: The filename of the log file. Use None to disable.
    '''
    
    if num_proc is not None:
        assert isinstance( num_proc, int ) and num_proc > 0, \
            f'num_proc must be a positive integer. Got {num_proc}. '

    # Find the folders and frames.
    depthfolderlist, framestrlist, framenum = find_folders_and_frames(trajdir, 'DepthPlanar', 'Depth')

    if num_proc is None:
        # Single process.
        with MeasureTime('SingleDepth', indent='    '):
            traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist = \
                single_verify_frame_dpeth( logf, trajdir, startind, framenum, framestrlist, depthfolderlist )
    else:
        # Multi-process.
        with MeasureTime('ParallelDepth', indent='    '):
            traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist = \
                parallel_verify_frame_dpeth( trajdir, startind, framenum, framestrlist, depthfolderlist, num_proc, log_file )

    return np.array(traj_dmean), np.array(traj_dmax), np.array(traj_dmin), np.array(traj_dstd), np.array(fileindlist)

#================================#
#============= RGB ==============#
#================================#

def proc_verify_frame_rgb(trajdir, rgbfolderlist, framestr):
    global PROC_IMG_READER, PROC_LOGGER
    
    dmax  = []
    dmin  = []
    dmean = []
    dstd  = []
    
    for _, rgbfolder in enumerate(rgbfolderlist):
        rgbmodstr = rgbfolder.split('image_')[-1] # hard coded
        rgbfile_surfix = rgbmodstr + '.png'
        rgbfile = join(trajdir, rgbfolder, framestr + '_' + rgbfile_surfix)
        
        if isfile(rgbfile):
            rgbnp = PROC_IMG_READER.read_rgb(rgbfile)
            dmax.append(  rgbnp.max()  )
            dmin.append(  rgbnp.min()  )
            dmean.append( rgbnp.mean() )
            dstd.append(  rgbnp.std()  )
        else:
            time_str = datetime.now().strftime("%m/%d/%y %H:%M:%S.%f")
            PROC_LOGGER.info(f'{time_str}: RGB file missing {rgbfile}. ')
    
    return dict(
        dmax=dmax, 
        dmin=dmin, 
        dmean=dmean, 
        dstd=dstd, 
        framestr=framestr )

def parallel_verify_frame_rgb(trajdir, startind, framenum, framestrlist, rgbfolderlist, num_proc, log_file):
    traj_dmax   = []
    traj_dmin   = []
    traj_dmean  = []
    traj_dstd   = []
    fileindlist = []
    
    # Prepare the arguments.
    framestr_list = [ framestrlist[k] for k in range(startind, framenum) ]
    N = len(framestr_list)
    rep_trajdir = ReplicatedArgument(trajdir, N)
    rep_rgbfolderlist = ReplicatedArgument(rgbfolderlist, N)
    
    zipped_args = zip( rep_trajdir, rep_rgbfolderlist, framestr_list )
    
    # Run in parallel.
    with PoolWithLogger( num_proc, proc_verify_frame_init, 'tartanair', log_file ) as pool:
        results = pool.map( proc_verify_frame_rgb, zipped_args )
        
        for res in results:
            traj_dmax.append(   res['dmax']     )
            traj_dmin.append(   res['dmin']     )
            traj_dmean.append(  res['dmean']    )
            traj_dstd.append(   res['dstd']     )
            fileindlist.append( res['framestr'] )
            
    return traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist

def single_verify_frame_rgb(logf, trajdir, startind, framenum, framestrlist, rgbfolderlist):
    imgreader = ImageReader()
    
    traj_dmax   = []
    traj_dmin   = []
    traj_dmean  = []
    traj_dstd   = []
    fileindlist = []

    for k in range(startind, framenum):
        framestr = framestrlist[k]
        dmax  = []
        dmin  = []
        dmean = []
        dstd  = []
        for w, rgbfolder in enumerate(rgbfolderlist):
            rgbmodstr = rgbfolder.split('image_')[-1] # hard coded
            rgbfile_surfix = rgbmodstr + '.png'

            rgbfile = join(trajdir, rgbfolder, framestr + '_' + rgbfile_surfix)
            if isfile(rgbfile):
                rgbnp = imgreader.read_rgb(rgbfile)
            else:
                logf.logline('RGB file missing ' + rgbfile)
                print ('RGB file missing', rgbfile)
                continue

            dmax.append(rgbnp.max())
            dmin.append(rgbnp.min())
            dmean.append(rgbnp.mean())
            dstd.append(rgbnp.std())

        traj_dmax.append(dmax)
        traj_dmin.append(dmin)
        traj_dmean.append(dmean)
        traj_dstd.append(dstd)
        fileindlist.append(framestr)

        if k%100==0:
            print("    Read {} RGB files...".format(k))
            
    print(f"    {framenum - startind} depth files processed. ")
            
    return traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist

def verify_traj_rgb( trajdir, logf, startind=0, num_proc=None, log_file=None):
    '''
    return rgb values in order to detect images too dark or too bright
    
    num_proc: number of processes. Use None to disable multiprocessing.
    log_file: The filename of the log file. Use None to disable.
    '''
    
    if num_proc is not None:
        assert isinstance( num_proc, int ) and num_proc > 0, \
            f'num_proc must be a positive integer. Got {num_proc}. '
    
    # Find the folders and frames.
    rgbfolderlist, framestrlist, framenum = find_folders_and_frames(trajdir, 'Scene', 'RGB')

    if num_proc is None:
        # Single process.
        with MeasureTime('SingleRGB', indent='    '):
            traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist = \
                single_verify_frame_rgb( logf, trajdir, startind, framenum, framestrlist, rgbfolderlist )
    else:
        # Multi-process.
        with MeasureTime('ParallelRGB', indent='    '):
            traj_dmax, traj_dmin, traj_dmean, traj_dstd, fileindlist = \
                parallel_verify_frame_rgb( trajdir, startind, framenum, framestrlist, rgbfolderlist, num_proc, log_file )

    return np.array(traj_dmean), np.array(traj_dmax), np.array(traj_dmin), np.array(traj_dstd), np.array(fileindlist)

#================================#
#======== Segmentation ==========#
#================================#

def proc_verify_frame_seg(trajdir, segfolderlist, framestr, color_dict, seglabels, segvalues):
    global PROC_IMG_READER, PROC_LOGGER
    
    segfoldernum = len(segfolderlist)
    
    pixelnum = {sl: np.zeros(segfoldernum, dtype=np.int32) for sl in seglabels} # each item is a K number list
    # errorlist = []
    # color_errorlist = []
    errordict_color_to_framestr = {}
    segdata = np.zeros((segfoldernum, 256), dtype=np.uint8) # 12 x 256 bool matrix recording the seg data in each frame
    
    for w, segfolder in enumerate(segfolderlist):
        segmodstr = segfolder.split('seg_')[-1] # hard coded
        segfile_surfix = segmodstr + '_seg.png'

        segfile = join(trajdir, segfolder, framestr + '_' + segfile_surfix)
        if isfile(segfile):
            segnp = PROC_IMG_READER.read_seg(segfile)
        else:
            PROC_LOGGER.info(f'Seg file missing {segfile}')
            continue

        segcolors = np.unique(segnp)
        for segcolor in segcolors: # enumerate all the labels in the image and calculate pixel number
            if segcolor not in color_dict: # move the printing after all the data are processed
            #     if segcolor not in color_errorlist: # not to print too much info
            #         PROC_LOGGER.info('Seg color {} in the seg image, but not in the color dict seg_rgbs.txt'.format(segcolor))
            #         color_errorlist.append(segcolor)
                errordict_color_to_framestr[segcolor] = framestr
                continue # this is for the zero color, which is not in the color_dict

            # tanslate color to index
            segind = color_dict[segcolor]
            segdata[w, segind] = 1

            if segind not in segvalues:
                # if segind not in errorlist: # not to print too much info
                #     PROC_LOGGER.info(f'{segfile}: Seg label error {segind} (color {segcolor}) in the seg image, but not in the env label file')
                #     errorlist.append(segind)
                errordict_color_to_framestr[segcolor] = framestr
                continue
            seglabel = segvalues[segind]
            # calculate the number of pixels 
            segnum = np.sum(segnp == segcolor)
            pixelnum[seglabel][w] = segnum
    
    return dict(
        pixelnum=pixelnum, 
        segdata = segdata,
        errordict=errordict_color_to_framestr )

def parallel_verify_frame_seg( trajdir, startind, framenum, framestrlist, segfolderlist, color_dict, seglabels, segvalues, num_proc, log_file ):
    pixelnums = {sl: [] for sl in seglabels} # each item is a N x K numpy array
    segdata_all = np.zeros((framenum-startind, len(segfolderlist), 256), dtype=np.uint8)
    errordict_all = {}
    # Prepare the arguments.
    framestr_list = [ framestrlist[k] for k in range(startind, framenum) ]
    N = len(framestr_list)
    rep_trajdir        = ReplicatedArgument(trajdir, N)
    rep_seghfolderlist = ReplicatedArgument(segfolderlist, N)
    rep_color_dict     = ReplicatedArgument(color_dict, N)
    rep_seglables      = ReplicatedArgument(seglabels, N)
    rep_segvalues      = ReplicatedArgument(segvalues, N)
    
    zipped_args = zip( rep_trajdir, rep_seghfolderlist, framestr_list, rep_color_dict, rep_seglables, rep_segvalues )
    
    # Run in parallel.
    with PoolWithLogger( num_proc, proc_verify_frame_init, 'tartanair', log_file ) as pool:
        results = pool.map( proc_verify_frame_seg, zipped_args )
        
        for k, res in enumerate(results):
            pixelnum = res['pixelnum']
            segdata = res['segdata']
            errordict = res['errordict']
            
            for sl in seglabels:
                pixelnums[sl].append(pixelnum[sl])

            segdata_all[k,:] = segdata

            for color, framestr in errordict.items():
                if color not in errordict_all:
                    errordict_all[color] = [framestr]
                else:
                    errordict_all[color].append(framestr)

    return pixelnums, segdata_all, errordict_all

def append_errorframe(errordict, color, framestr):
    if color not in errordict:
        errordict[color] = [framestr]
    else:
        if framestr not in errordict[color]:
            errordict[color].append(framestr)
    return errordict

def single_verify_frame_seg(logf, trajdir, startind, framenum, framestrlist, segfolderlist, color_dict, seglabels, segvalues):
    '''
    Return 
        pixelnums: dictionary {seg-class: N x 12 numpy array} 
        segdata: N x 12 x 256 array, whether a seg class appears in one frame
        errordict_color_to_framelist: dictionary {seg-color: [frameind0, frameind1, ...]} where seg-color is with problem
    '''
    imgreader = ImageReader()
    segfoldernum = len(segfolderlist)
    
    pixelnums = {sl: [] for sl in seglabels} # each item is a N x K numpy array
    errordict_color_to_framelist = {}
    seg_data_all = []
    for k in range(startind, framenum):
        framestr = framestrlist[k]
        pixelnum = {sl: np.zeros(segfoldernum, dtype=np.int32) for sl in seglabels} # each item is a K number list
        segdata = np.zeros((len(segfolderlist), 256), dtype=np.uint8) # 12 x 256 bool matrix recording the seg data in each frame
        for w, segfolder in enumerate(segfolderlist):
            segmodstr = segfolder.split('seg_')[-1] # hard coded
            segfile_surfix = segmodstr + '_seg.png'

            segfile = join(trajdir, segfolder, framestr + '_' + segfile_surfix)
            if isfile(segfile):
                segnp = imgreader.read_seg(segfile)
            else:
                logf.logline('Seg file missing ' + segfile)
                print ('Seg file missing', segfile)
                continue

            segcolors = np.unique(segnp)
            for segcolor in segcolors: # enumerate all the labels in the image and calculate pixel number
                # according to seg_rgbs.txt color 43, 55 are missing in the first column
                if segcolor not in color_dict: # move the printing after all data are processed
                    # if segcolor not in color_errorlist: # not to print too much info
                    #     logf.logline('Seg color {} in the seg image, but not in the env label file'.format(segcolor))
                    #     print ('Seg color {} in the seg image, but not in the env label file'.format(segcolor))
                    #     color_errorlist.append(segcolor)
                    errordict_color_to_framelist = append_errorframe(errordict_color_to_framelist, segcolor, framestr)
                    continue

                # tanslate color to index
                segind = color_dict[segcolor]
                segdata[w, segind] = 1

                if segind not in segvalues:
                    # if segind not in errorlist: # not to print too much info
                    #     logf.logline('Seg label error {} (color {}) in the seg image, but not in the env label file'.format(segind, segcolor))
                    #     print ('Seg label error {} (color {}) in the seg image, but not in the env label file'.format(segind, segcolor))
                    #     errorlist.append(segind)
                    errordict_color_to_framelist = append_errorframe(errordict_color_to_framelist, segcolor, framestr)
                    continue
                seglabel = segvalues[segind]
                # calculate the number of pixels 
                segnum = np.sum(segnp == segcolor)
                pixelnum[seglabel][w] = segnum


        seg_data_all.append(segdata)
        for sl in seglabels:
            pixelnums[sl].append(pixelnum[sl])

        if k%100==0:
            print("    Read {} Seg files...".format(k))
    
    return pixelnums, np.array(seg_data_all), errordict_color_to_framelist

def get_seg_color_label_value(labelfile):
    global _CURRENT_PATH
    
    seg_colors = np.loadtxt(_CURRENT_PATH + '/seg_rgbs.txt', delimiter=',', dtype=np.uint8)
    color_dict = {seg_colors[k, 2]: k for k in range(1, len(seg_colors)-1)}

    with open(labelfile,'r') as f:
        seglabels = json.load(f)
        seglabels = seglabels["name_map"] # {name: ind}
        segvalues = {seglabels[lab]:lab for lab in seglabels} # {ind: name}
        
    return color_dict, seglabels, segvalues

def verify_traj_seg(trajdir, labelfile, logf, startind=0, num_proc=None, log_file=None):
    '''
    1. verify the seg values are consistent with the label file
    2. calculate the percentage of each class
    
    num_proc: number of processes. Use None to disable multiprocessing.
    log_file: The filename of the log file. Use None to disable.

    return 
        pixelnums: a dictionary class-label: N x K numpy array, pixel number 
        segdata: N x 12 x 256 array, whether a seg class appears in one frame
    '''
    
    if num_proc is not None:
        assert isinstance( num_proc, int ) and num_proc > 0, \
            f'num_proc must be a positive integer. Got {num_proc}. '
    
    # Find the folders and frames.
    segfolderlist, framestrlist, framenum = find_folders_and_frames(trajdir, 'Segmentation', 'Seg')
    
    # Find the color dict, seg labels, and seg values.
    color_dict, seglabels, segvalues = get_seg_color_label_value(labelfile)

    if num_proc is None:
        # Single process.
        with MeasureTime('SingleSeg', indent='    '):
            pixelnums, segdata, errordict = \
                single_verify_frame_seg(
                    logf, trajdir, startind, framenum, framestrlist, segfolderlist, color_dict, seglabels, segvalues)
    else:
        # Multi-process.
        with MeasureTime('ParallelSeg', indent='    '):
            pixelnums, segdata, errordict = \
                parallel_verify_frame_seg(
                    trajdir, startind, framenum, framestrlist, segfolderlist, color_dict, seglabels, segvalues, num_proc, log_file)

    for sl in seglabels:
        pixelnums[sl] = np.array(pixelnums[sl])
    
    return pixelnums, segdata, errordict

if __name__=="__main__":
    from .data_validation import FileLogger
    trajdir = 'E:\\TartanAir_v2\\WaterMillDayExposure\\Data_easy\\P000'
    logf = FileLogger('..\\testlog.txt')
    # traj_dmean, traj_dmax, traj_dmin, traj_dstd, fileindlist = verify_traj_depth(trajdir, logf)

    # traj_dmean, traj_dmax, traj_dmin, traj_dstd, fileindlist = verify_traj_rgb(trajdir, logf)

    labelfile = "E:\\TartanAir_v2\\WaterMillDayExposure\\seg_label.json"
    pixelnums, segdata = verify_traj_seg(trajdir, labelfile, logf = logf, num_proc=8, log_file='..\\testlog_mp.txt')

    import ipdb;ipdb.set_trace()
