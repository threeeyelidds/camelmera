'''
Initialize the viewing camera's parameters of Open3D window 
Manually review the point cloud visualization frame by frame

Usage:
1. Set init to True to initialize the camera parameters
2. A window will pop up where one can arbitrarily adjust the viewing angle,
   zoom in/out, rotate the point cloud, etc.
3. Press 'q' to finish initialization and the camera parameters are saved
4. A window will pop up visualizing the point clouds frame by frame with
   the camera parameters just saved
5. Press left-arrow key to move back one frame and right-arrow key to move
   to the next frame. Press 'q' to exit the visualization
6. Use the saved camera parameters in postprocessing.lidar_visualization
'''

import numpy as np
import open3d as o3d
from glob import glob

def vis(input_dir, init=True):

    frames = sorted(glob(input_dir + '/*.ply'))

    if init:
        vis0 = o3d.visualization.Visualizer()
        vis0.create_window(window_name='Set_Camera_Param', width=1920, height=480)

        pcd = o3d.io.read_point_cloud(frames[0])
        vis0.add_geometry(pcd)
        vis0.run()  # change the view and press 'q' to terminate
        param = vis0.get_view_control().convert_to_pinhole_camera_parameters()
        vis0.destroy_window()

        np.savez('o3d_camera.npz', w=param.intrinsic.width, 
                                h=param.intrinsic.height, 
                                intrinsic=param.intrinsic.intrinsic_matrix,
                                extrinsic=param.extrinsic)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Visualize PCD', width=1920, height=480)

    idx = 0
    pcd = o3d.io.read_point_cloud(frames[idx])
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    
    cam = o3d.camera.PinholeCameraParameters()
    camerafile = np.load('o3d_camera.npz')
    w, h = camerafile['w'], camerafile['h']
    intr_mat, ext_mat = camerafile['intrinsic'], camerafile['extrinsic']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, 
                    intr_mat[0,0], intr_mat[1,1], 
                    intr_mat[0,-1], intr_mat[1,-1])
    intrinsic.intrinsic_matrix = intr_mat
    cam.intrinsic = intrinsic
    cam.extrinsic = ext_mat
    ctr.convert_from_pinhole_camera_parameters(cam)

    def right_click(vis):
        nonlocal idx
        if idx + 1 >= len(frames):
            return
        idx = idx + 1
        vis.clear_geometries()
        pcd = o3d.io.read_point_cloud(frames[idx])
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(cam)
    
    def left_click(vis):
        nonlocal idx
        if idx - 1 < 0:
            return
        idx = idx - 1
        vis.clear_geometries()
        pcd = o3d.io.read_point_cloud(frames[idx])
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(cam)
        
    def exit_key(vis):
        vis.destroy_window()
    
    vis.register_key_callback(262, right_click)
    vis.register_key_callback(263, left_click)
    vis.register_key_callback(32, exit_key)
    vis.poll_events()
    vis.run()


if __name__=='__main__':
    vis('/home/shihao/ImageFlow-Dataset/Dummy_env/Data_easy/P000/lidar', init=False)