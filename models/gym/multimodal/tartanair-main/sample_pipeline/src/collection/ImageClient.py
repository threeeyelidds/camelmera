import airsim
from airsim.types import Pose, Vector3r, Quaternionr

import cv2 # debug
import numpy as np

# from PanoramaDepth2Distance import (
#     meshgrid_from_img, depth_2_distance, cuda_depth_2_distance )

np.set_printoptions(precision=3, suppress=True, threshold=10000)

  # Scene = 0, 
  # DepthPlanar = 1, 
  # DepthPerspective = 2,
  # DepthVis = 3, 
  # DisparityNormalized = 4,
  # Segmentation = 5,
  # SurfaceNormals = 6,
  # Infrared = 7
class ImageClient(object):
    def __init__(self, camlist, typelist, ip=''):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()

        self.IMGTYPELIST = typelist
        self.CAMLIST = camlist

        self.imgRequest = []
        for k in self.CAMLIST:
            for imgtype in self.IMGTYPELIST:
                if imgtype == 'Scene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Scene, False, False))

                elif imgtype == 'DepthPlanar':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPlanar, True))

                elif imgtype == 'DepthPerspective': # for debug
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPerspective, True))

                elif imgtype == 'Segmentation':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Segmentation, False, False))

                elif imgtype == 'CubeScene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.CubeScene, False, True))

                elif imgtype == 'CubeDistance':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.CubeDepth, True, False))

                else:
                    print ('Error image type: {}'.format(imgtype))

        # Meshtrid coordinates for the panorama images.
        # Will be populated upon receiving the first panorama image.
        self.panorama_xx = None
        self.panorama_yy = None
        self.cube_cuda = True # Set True to use CUDA.

    def get_cam_pose(self, response):
        cam_pos = response.camera_position # Vector3r
        cam_ori = response.camera_orientation # Quaternionr

        cam_pos_vec = [cam_pos.x_val, cam_pos.y_val, cam_pos.z_val]
        cam_ori_vec = [cam_ori.x_val, cam_ori.y_val, cam_ori.z_val, cam_ori.w_val]

        # print cam_pos_vec, cam_ori_vec
        return cam_pos_vec + cam_ori_vec

    def readimgs(self):
        # responses = self.client.simGetImages(self.imgRequest) # discard the first query because of AirSim error
        responses = self.client.simGetImages(self.imgRequest)
        camposelist = []
        data_dict = {}
        # rgblist, depthlist, seglist = [], [], []
        # rgblist_cube, distlist_cube = [], []
        idx = 0
        for k, cam in enumerate(self.CAMLIST):
            for imgtype in self.IMGTYPELIST:
                response = responses[idx]
                hh, ww = response.height, response.width
                if hh==0 or ww==0:
                    print ('Error read image: {}'.format(imgtype))
                    return None, None
                # response_nsec = response.time_stamp
                # response_time = rospy.rostime.Time(int(response_nsec/1000000000),response_nsec%1000000000)
                if imgtype == 'DepthPlanar': #response.pixels_as_float:  # for depth data
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    dataimg = img1d.reshape(hh, ww)
                    # depthlist.append(depthimg)

                elif imgtype == 'DepthPerspective': #for debug
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    dataimg = img1d.reshape(hh, ww)
                    # depthlist.append(depthimg)

                elif imgtype == 'Scene':  # raw image data
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                    dataimg = img1d.reshape(hh, ww, -1)
                    # rgblist.append(rgbimg[:,:,:3])

                elif imgtype == 'Segmentation': # TODO: should map the RGB back to index
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgba = img1d.reshape(hh, ww, -1)
                    # import ipdb;ipdb.set_trace()
                    dataimg = img_rgba[:,:,0]
                    # seglist.append(img_seg)

                elif imgtype == 'CubeScene':
                    # Decode the image directly from the bytes.
                    decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)
                    dataimg = decoded[:, :, :3]
                    # rgblist_cube.append(decoded[:, :, :3])

                elif imgtype == 'CubeDistance':
                    # Get a PFM format array.
                    pfm = np.reshape(
                        np.asarray(response.image_data_float, np.float32), 
                        (hh, ww))
                    
                    # Convert the depth image to distance image.
                    dataimg = np.zeros_like(pfm)
                    if ( self.cube_cuda ):
                        # if ( not cuda_depth_2_distance(pfm, dist) ):
                        #     raise Exception('Failed to convert cube depth image to distance image (CUDA). ')
                        pass
                    else:
                        # Meshgrid coordinates.
                        # if ( self.panorama_xx is None or self.panorama_yy is None ):
                        #     self.panorama_yy, self.panorama_yy = \
                        #         meshgrid_from_img(pfm)

                        # if ( not depth_2_distance(pfm, self.panorama_xx, self.panorama_yy, dist) ):
                        #     raise Exception('Failed to convert cube depth image to distance image (CPU)')
                        pass
                    # distlist_cube.append(dist)

                datakey = imgtype + '_' + cam # key: TYPE_CAM   
                data_dict[datakey] = dataimg
                idx += 1
            # import ipdb;ipdb.set_trace()
            cam_pose_img = self.get_cam_pose(response) # get the cam pose for each camera
            camposelist.append(cam_pose_img)

        return data_dict, camposelist

    def listobjs(self):
        object_list = sorted(self.client.simListSceneObjects())
        object_seg_ids = [self.client.simGetSegmentationObjectID(object_name.lower()) for object_name in object_list]

        for object_name, object_seg_id in zip(object_list, object_seg_ids):
            if object_seg_id!=-1:
                print("object_name: {}, object_seg_id: {}".format(object_name, object_seg_id))


    def setpose(self, pose):
        self.client.simSetVehiclePose(pose, ignore_collision=True)

    def getpose(self):
        return self.client.simGetVehiclePose()

    def simPause(self, pause): # this is valid for customized AirSim
        return self.client.simPause(pause)

    def close(self):
        self.client.simPause(False)
        self.client.reset()
