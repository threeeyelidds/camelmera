import numpy as np
import cv2

from .ImageReader import ImageReader
from .data_enumeration import enumerate_modalities, enumerate_frames
from os.path import isdir, join, split

import os
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

class DataVisualizer(object):
    def __init__(self) -> None:
        self.seg_colors = np.loadtxt(_CURRENT_PATH + '/seg_rgbs.txt', delimiter=',',dtype=np.uint8)
        self.text_bg_color = (230, 130, 10) # BGR
        self.text_color = (70, 200, 230)

    def calculate_angle_distance_from_du_dv(self, du, dv, flagDegree=False):
        a = np.arctan2( dv, du )

        angleShift = np.pi

        if ( True == flagDegree ):
            a = a / np.pi * 180
            angleShift = 180
            # print("Convert angle from radian to degree as demanded by the input file.")

        d = np.sqrt( du * du + dv * dv )

        return a, d, angleShift

    def visflow(self, flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
        """
        Show a optical flow field as the KITTI dataset does.
        Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
        """

        ang, mag, _ = self.calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

        # Use Hue, Saturation, Value colour model 
        hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

        am = ang < 0
        ang[am] = ang[am] + np.pi * 2

        hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
        hsv[ :, :, 1 ] = mag / maxF * n
        hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

        hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
        hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
        hsv = hsv.astype(np.uint8)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if ( mask is not None ):
            mask = mask != 255
            bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

        return bgr


    def visdepth(self, depth):
        depthvis = np.clip(400/(depth+1e-6) ,0 ,255)
        depthvis = depthvis.astype(np.uint8)
        depthvis = cv2.applyColorMap(depthvis, cv2.COLORMAP_JET)

        return depthvis

    def visdisparity(self, disp, maxthresh = 50):
        dispvis = np.clip(disp,0,maxthresh)
        dispvis = dispvis/maxthresh*255
        dispvis = dispvis.astype(np.uint8)
        dispvis = cv2.applyColorMap(dispvis, cv2.COLORMAP_JET)

        return dispvis

    def visseg(self, segnp):
        segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

        # for k in range(256):
        #     mask = segnp==k
        #     colorind = k % len(colors)
        #     if np.sum(mask)>0:
        #         segvis[mask,:] = colors[colorind]

        segvis = self.seg_colors[ segnp, : ]
        segvis = segvis.reshape( segnp.shape+(3,) )

        return segvis

    def add_text(self, img, text, offset_height = 0):
        textlen = len(text)
        bg_width = textlen * 10
        bg_height = 30
        x, y = 10 + offset_height , 0
        
        img[x:x+bg_height, y:y+bg_width, :] = img[x:x+bg_height, y:y+bg_width, :] * 0.5 + np.array(self.text_bg_color) * 0.5
        cv2.putText(img,text,(y+10, x + 5 + bg_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, thickness=1)

        return img

def vis_frame(imgreader, imgvisualizer, trajdir, framestr, scale, depthfolderlist = None, rgbfolderlist = None, segfolderlist = None):
    visimgs = []
    for w in [0,2,1,3]: # front, right, back, left
        visimg = None
        if rgbfolderlist is not None:
            rgbmodstr = rgbfolderlist[w].split('image_')[-1] # hard coded
            rgbfile_surfix = rgbmodstr + '.png'
            rgbfile = join(trajdir, rgbfolderlist[w], framestr + '_' + rgbfile_surfix)
            rgbnp = imgreader.read_rgb(rgbfile)
            rgbvis = cv2.resize(rgbnp, (0,0), fx=scale, fy=scale)
            visimg = rgbvis if visimg is None else np.concatenate((visimg, rgbvis), axis = 0)

        if depthfolderlist is not None:
            depthmodstr = depthfolderlist[w].split('depth_')[-1] # hard coded
            depthfile_surfix = depthmodstr + '_depth.png'
            depthfile = join(trajdir, depthfolderlist[w], framestr + '_' + depthfile_surfix)
            depthnp = imgreader.read_disparity(depthfile)
            depthvis = imgvisualizer.visdisparity(depthnp)
            depthvis = cv2.resize(depthvis, (0,0), fx=scale, fy=scale)
            visimg = depthvis if visimg is None else np.concatenate((visimg, depthvis), axis = 0)

        if segfolderlist is not None:
            segmodstr = segfolderlist[w].split('seg_')[-1] # hard coded
            segfile_surfix = segmodstr + '_seg.png'
            segfile = join(trajdir, segfolderlist[w], framestr + '_' + segfile_surfix)
            segnp = imgreader.read_seg(segfile)
            segvis = imgvisualizer.visseg(segnp)
            segvis = cv2.resize(segvis, (0,0), fx=scale, fy=scale)
            visimg = segvis if visimg is None else np.concatenate((visimg, segvis), axis = 0)

        visimgs.append(visimg)

    if len(visimgs) == 0:
        return None 
    visimgs = np.concatenate(visimgs, axis=1)
    return visimgs

def save_vid_with_statistics(args): 
    '''
    process one trajectory and output a vedio file
    put statistics values on the frames
    outvidfile: xxx.mp4
    scale: scale the image in the video
    startind: the image index does not start from 0
    check_depth: put text on the image about the depth statistics
    '''
    trajdir, outvidfolder = args
    scale=0.25
    startind=0
    imgw, imgh = 640, 640
    imgreader = ImageReader()
    imgvisualizer = DataVisualizer()
    # find the folders
    modfolder_dict = enumerate_modalities(trajdir)
    depthfolderlist = modfolder_dict['DepthPlanar'] # hard coded, need to change for new airsim version
    rgbfolderlist = modfolder_dict['Scene'] # hard coded, need to change for new airsim version
    segfolderlist = modfolder_dict['Segmentation'] # hard coded, need to change for new airsim version

    framestrlist = enumerate_frames(join(trajdir, depthfolderlist[0]))
    framestrlist.sort()
    framenum = len(framestrlist)

    tempstrs, trajstr = split(trajdir)
    _, datastr = split(tempstrs)
    anafolder = join(trajdir.split(datastr)[0], 'analyze')

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    outvidfile = join(outvidfolder, datastr + '_' + trajstr + '.mp4')
    fout=cv2.VideoWriter(outvidfile, fourcc, 10.0, (int(imgw*4*scale), int(imgh*3*scale)))

    if isdir(anafolder):
        # read statistic files from the analyze folder, and put it on the image
        dispmaxfile_traj = join(anafolder, datastr+'_'+trajstr+'_disp_max.npy')
        dispminfile_traj = join(anafolder, datastr+'_'+trajstr+'_disp_min.npy')
        dispmeanfile_traj = join(anafolder, datastr+'_'+trajstr+'_disp_mean.npy')
        dispstdfile_traj = join(anafolder, datastr+'_'+trajstr+'_disp_std.npy')
        dispmax_traj = np.load(dispmaxfile_traj)
        dispmin_traj = np.load(dispminfile_traj)
        dispmean_traj = np.load(dispmeanfile_traj)
        dispstd_traj = np.load(dispstdfile_traj)

        rgbmaxfile_traj = join(anafolder, datastr+'_'+trajstr+'_rgb_max.npy')
        rgbminfile_traj = join(anafolder, datastr+'_'+trajstr+'_rgb_min.npy')
        rgbmeanfile_traj = join(anafolder, datastr+'_'+trajstr+'_rgb_mean.npy')
        rgbstdfile_traj = join(anafolder, datastr+'_'+trajstr+'_rgb_std.npy')
        rgbmax_traj = np.load(rgbmaxfile_traj)
        rgbmin_traj = np.load(rgbminfile_traj)
        rgbmean_traj = np.load(rgbmeanfile_traj)
        rgbstd_traj = np.load(rgbstdfile_traj)

        # segfile_traj = join(anafolder, datastr+'_'+trajstr+'_seg.npy')

    for k in range(startind, framenum):
        framestr = framestrlist[k]
        visimgs = vis_frame(imgreader, imgvisualizer, trajdir, framestr, scale, depthfolderlist, rgbfolderlist, segfolderlist)

        if isdir(anafolder):
            # read statistic files from the analyze folder, and put it on the image
            dispmax = dispmax_traj[k].mean()
            dispmin = dispmin_traj[k].mean()
            dispmean = dispmean_traj[k].mean()
            dispstd = dispstd_traj[k].mean()

            rgbmax = rgbmax_traj[k].mean()
            rgbmin = rgbmin_traj[k].mean()
            rgbmean = rgbmean_traj[k].mean()
            rgbstd = rgbstd_traj[k].mean()

            # segpixels = segfile_traj[k]
            depthtext = '{}. Disparity mean={:.2f}, std={:.2f}, max={:.2f}, min={:.2f}'.format(str(k).zfill(4), \
                                dispmean, dispstd, dispmax, dispmin)
            rgbtext = '{}. RGB mean={:.2f}, std={:.2f}, max={:.2f}, min={:.2f}'.format(str(k).zfill(4), \
                                rgbmean, rgbstd, rgbmax, rgbmin)

            visimgs = imgvisualizer.add_text(visimgs, rgbtext)
            visimgs = imgvisualizer.add_text(visimgs, depthtext, offset_height=int(imgh*scale))

        fout.write(visimgs)
        # cv2.imshow('img', visimgs)
        # cv2.waitKey(0)
    fout.release()


if __name__=="__main__":
    # dv = DataVisualizer()
    # img = cv2.imread('/home/amigo/tmp/test_sample_trajs/Data/P000_smooth10/image_front/000013_front.png')
    # img = dv.add_text(img, 'test the text')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # from os import mkdir
    # trajdir = '/home/amigo/tmp/test_root/coalmine/Data_easy/P000'
    # outvidfolder = '/home/amigo/tmp/test_root/coalmine/video'
    # if not isdir(outvidfolder):
    #     mkdir(outvidfolder)

    # save_vid_with_statistics(trajdir, outvidfolder, scale=0.25, startind=0)

    trajdir = 'E:\\TartanAir_v2\\OldScandinaviaExposure\\Data_easy\\P000'
    imgreader = ImageReader()
    imgvisualizer = DataVisualizer()

    framestr = '000000'
    scale = 0.5
    depthfolderlist = [
        'depth_lcam_front',
        'depth_lcam_back',
        'depth_lcam_right',
        'depth_lcam_left',
        'depth_lcam_top',
        'depth_lcam_bottom'
    ]
    rgbfolderlist = [
        'image_lcam_front',
        'image_lcam_back',
        'image_lcam_right',
        'image_lcam_left',
        'image_lcam_top',
        'image_lcam_bottom'
    ]
    segfolderlist = [
        'seg_lcam_front',
        'seg_lcam_back',
        'seg_lcam_right',
        'seg_lcam_left',
        'seg_lcam_top',
        'seg_lcam_bottom'
    ]
    vis = vis_frame(imgreader, imgvisualizer, trajdir, framestr, scale, depthfolderlist, rgbfolderlist, segfolderlist)
    cv2.imshow('img',vis)
    cv2.waitKey(0)