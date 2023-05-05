import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from glob import glob
from .data_visualization import DataVisualizer
from .ImageReader import ImageReader
from .data_enumeration import enumerate_modalities, enumerate_frames
from os.path import isdir, join, split

def vispcd( pcd, o3d_cam=None):
    w, h =  (1920, 480)  # default o3d window size

    if o3d_cam:
        camerafile = o3d_cam
        w, h = camerafile['w'], camerafile['h']
        cam = o3d.camera.PinholeCameraParameters()
        
        intr_mat, ext_mat = camerafile['intrinsic'], camerafile['extrinsic']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, 
                        intr_mat[0,0], intr_mat[1,1], 
                        intr_mat[0,-1], intr_mat[1,-1])
        intrinsic.intrinsic_matrix = intr_mat
        cam.intrinsic = intrinsic
        cam.extrinsic = ext_mat

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(pcd)

    if o3d_cam:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    vis.poll_events()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img = np.array(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def save_vid(trajdir, outvidfolder, scale=0.25, startind=0, o3d_cam=None): 
    '''
    process one trajectory and output a vedio file
    put statistics values on the frames
    outvidfile: xxx.mp4
    scale: scale the image in the video
    startind: the image index does not start from 0
    o3d_cam: the viewing camera's parameters of Open3D window
    '''
    imgw, imgh = 640, 640
    imgreader = ImageReader()
    imgvisualizer = DataVisualizer()
    # find the folders
    modfolder_dict = enumerate_modalities(trajdir)
    depthfolderlist = modfolder_dict['DepthPlanar'] # hard coded, need to change for new airsim version
    rgbfolderlist = modfolder_dict['Scene'] # hard coded, need to change for new airsim version
    lidarfolder = 'lidar'

    framestrlist = enumerate_frames(join(trajdir, depthfolderlist[0]))
    framestrlist.sort()
    framenum = len(framestrlist)

    tempstrs, trajstr = split(trajdir)
    _, datastr = split(tempstrs)

    outvidfile = join(outvidfolder, datastr + '_' + trajstr + '_lidar.mp4')
    fout = cv2.VideoWriter(outvidfile, cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(imgw*4*scale), int(imgh*3*scale))))

    for k in range(startind, framenum):
        framestr = framestrlist[k]
        visimgs = []

        lidarmodstr = depthfolderlist[0].split('depth_')[-1] # hard coded
        lidarfile_surfix = lidarmodstr + '_lidar.ply'
        lidarfile = join(trajdir, lidarfolder, framestr + '_' + lidarfile_surfix)
        pcd = o3d.io.read_point_cloud(lidarfile)
        pcdimg = vispcd(pcd, o3d_cam=o3d_cam)
        pcdimg = cv2.resize(pcdimg, (imgw*4, imgh), interpolation = cv2.INTER_AREA)
        pcdvis = cv2.resize(pcdimg, (0,0), fx=scale, fy=scale)
        
        for w in [0,2,1,3]: # front, right, back, left
            depthmodstr = depthfolderlist[w].split('depth_')[-1] # hard coded
            depthfile_surfix = depthmodstr + '_depth.png'
            depthfile = join(trajdir, depthfolderlist[w], framestr + '_' + depthfile_surfix)
            depthnp = imgreader.read_disparity(depthfile)
            depthvis = imgvisualizer.visdisparity(depthnp)
            depthvis = cv2.resize(depthvis, (0,0), fx=scale, fy=scale)

            rgbmodstr = rgbfolderlist[w].split('image_')[-1] # hard coded
            rgbfile_surfix = rgbmodstr + '.png'
            rgbfile = join(trajdir, rgbfolderlist[w], framestr + '_' + rgbfile_surfix)
            rgbnp = imgreader.read_rgb(rgbfile)
            rgbvis = cv2.resize(rgbnp, (0,0), fx=scale, fy=scale)

            visimg = np.concatenate((rgbvis, depthvis), axis=0)
            visimgs.append(visimg)

        visimgs = np.concatenate(visimgs, axis=1)
        visimgs = np.concatenate((visimgs, pcdvis), axis=0)
        fout.write(visimgs)
        # cv2.imshow('img', visimgs)
        # cv2.waitKey(10)
    fout.release()

if __name__=='__main__':
    from os import mkdir
    trajdir = r'E:\\AbandonedSchoolExposure\\Data_easy\\P000'
    outvidfolder = r'D:\\test_data'
    o3d_cam = r'C:\\programs\\tartanair\\sample_pipeline\\src\\postprocessing\\o3d_camera.npz'
    o3d_cam_np = np.load(o3d_cam)

    if not isdir(outvidfolder):
        mkdir(outvidfolder)

    save_vid(trajdir, outvidfolder, scale=0.25, startind=0, o3d_cam=o3d_cam_np)