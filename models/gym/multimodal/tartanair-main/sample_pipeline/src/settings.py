import argparse

def get_args():
    parser = argparse.ArgumentParser(description='sample_pipeline')

    # pipeline control
    parser.add_argument('--mapping', action='store_true', default=False,
                        help='mapping the environment (default: False)')

    parser.add_argument('--sample-graph', action='store_true', default=False,
                        help='sample graph (default: False)')

    parser.add_argument('--sample-path', action='store_true', default=False,
                        help='sample path (default: False)')

    parser.add_argument('--sample-position', action='store_true', default=False,
                        help='sample position (default: False)')

    parser.add_argument('--data-collection', action='store_true', default=False,
                        help='data collection (default: False)')

    # mapping - expo_control
    parser.add_argument('--map-dir', default='~/tmp',
                        help='output map file directory')

    parser.add_argument('--map-filename', default='map',
                        help='output map file name')

    parser.add_argument('--path-skip', type=int, default=7,
                        help='skip steps on the path, the bigger the faster (default: 7)')

    parser.add_argument('--global-only', action='store_true', default=False,
                        help='only visit global frontiers, fast but inaccurate (default: False)')

    parser.add_argument('--camera-fov', type=float, default=90.0,
                        help='camera fov (default: 100)')

    parser.add_argument('--far-point', type=int, default=22,
                        help='a number a little larger than the Lidar range (default: 22)')

    parser.add_argument('--try-round', type=int, default=-1,
                        help='A* planning round (default: unlimited)')


    # Trajectory sampling 
    parser.add_argument('--max-failure-num', type=int, default=20,
                        help='maximum number of planning failure before giveup (default: 20)')


    # image collection - collect_images
    parser.add_argument('--environment-dir', default='',
                        help='root folder of one env for the trajectory data')

    parser.add_argument('--posefile-folder', default='',
                        help='folder for pose files (default: )')

    parser.add_argument('--data-folder', default='Data',
                        help='output folder for the trajectories (default: )')

    parser.add_argument('--cam-list', default='1_2',
                        help='camera list: 0-front, 1-right, 2-left, 3-back, 4-bottom (default: 1_2)')

    parser.add_argument('--img-type', default='Scene_DepthPlanar_Segmentation',
                        help='image type Scene, DepthPlanar, Segmentation (default: Scene_DepthPlanar_Segmentation)')

    parser.add_argument('--load-existing-trajectories', action='store_true', default=False,
                        help='load both position and orientation from file')

    parser.add_argument('--gamma', type=float, default=3.7,
                        help='gamma in airsim settings.json')

    parser.add_argument('--min-exposure', type=float, default=0.3,
                        help='MinExposure in airsim settings.json')

    parser.add_argument('--max-exposure', type=float, default=0.7,
                        help='MaxExposure in airsim settings.json')

    parser.add_argument('--save-posefile-only', action='store_true', default=False,
                        help='only sample the poses without save the image data')

    parser.add_argument('--airsim-ip', type=str, default='127.0.0.1', 
                        help='The IP address of the AirSim instance. ')

    parser.add_argument('--disable-cube-cuda', action='store_true', default=False, 
                        help='Disable CUDA when computing the cube distance image.')

    parser.add_argument('--traj-folder', default='',
                        help='folder for the fexisting trajectories, used when --load-existing-trajectories (default: )')

    parser.add_argument('--posefile-name', default='',
                        help='posefile used for data collection, used when --load-existing-trajectories (default: )')

    parser.add_argument('--dyna-time', type=float, default=0.0,
                        help='for dynamic environments, sleep this time between two frames')

    parser.add_argument('--traj-overwrite', action='store_true', default=False,
                        help='overwrite existing trajectory without create a separate folder with timestr surfix')

    parser.add_argument('--max-framenum', type=int, default=0,
                        help='the maximum number of frames to collect for each trajectory, set to 0 if no maximun limit')

    # image validation
    parser.add_argument('--data-root', default='',
                        help='root data folder that contrains environment folders')

    parser.add_argument('--env-folders', default='',
                        help='specify the environment folder, all the folders if not specified')

    parser.add_argument('--data-folders', default='Data_easy,Data_hard',
                        help='data folders in each environment folder')

    parser.add_argument('--traj-folders', default='',
                        help='trajecory folders to be processed')

    parser.add_argument('--create-video', action='store_true', default=False,
                        help='generate preview video (default: False)')

    parser.add_argument('--video-with-flow', action='store_true', default=False,
                        help='save flow in the video instead of right image (default: False)')

    parser.add_argument('--analyze-depth', action='store_true', default=False,
                        help='calculate depth statistics from depth image and output files (default: False)')

    parser.add_argument('--depth-from-file', action='store_true', default=False,
                        help='read depth info from files, only active when --analyze-depth is set (default: False)')

    parser.add_argument('--analyze-rgb', action='store_true', default=False,
                        help='read rgb image and output statistics file (default: False)')

    parser.add_argument('--seg-label-file', default='seg_label.json',
                        help='label file for segmentation (default: False)')

    parser.add_argument('--analyze-seg', action='store_true', default=False,
                        help='calculate seg statistics from seg image and output files (default: False)')

    parser.add_argument('--depth-filter', action='store_true', default=False,
                        help='filter depth and generate text file for stereo training (default: False)')

    parser.add_argument('--rgb-depth-filter', action='store_true', default=False,
                        help='filter depth and rgb value and generate text file for stereo training (default: False)')

    parser.add_argument('--update-ana-files', action='store_true', default=False,
                        help='after removing the trajs, update the analysis files (default: False)')

    # multi-process
    parser.add_argument("--np", type=int, default=1, 
                        help="Number of processes.")

    # for optical flow generation
    parser.add_argument("--index-step", type=int, default=1, 
                        help="Generate optical flow for every STEP ")

    parser.add_argument("--start-index", type=int, default=0, 
                        help="Skip the first few images ")

    parser.add_argument("--focal", type=int, default=320, 
                        help="camera focal length")

    parser.add_argument("--image-width", type=int, default=640, 
                        help="image width")

    parser.add_argument("--image-height", type=int, default=640, 
                        help="image height")

    parser.add_argument('--save-flow-image', action='store_true', default=False,
                        help='save optical flow image for debugging')

    parser.add_argument('--force-overwrite', action='store_true', default=False,
                        help='save optical flow in a same flow folder')

    parser.add_argument('--flow-outdir', default='flow_lcam_front',
                        help='output flow file to this folder (default: flow)')

    parser.add_argument('--target-root', default='',
                        help='copying flow to another drive, target root data folder that contrains environment folders')

    parser.add_argument('--img-folder', default='image_lcam_front',
                        help='input image folder (default: image_lcam_front)')

    parser.add_argument('--img-suffix', default='_lcam_front.png',
                        help='input image file name suffix (default: _lcam_front.png)')

    parser.add_argument('--depth-folder', default='depth_lcam_front',
                        help='input depth folder (default: image_lcam_front)')

    parser.add_argument('--depth-suffix', default='_lcam_front_depth.png',
                        help='input depth file name suffix (default: _lcam_front_depth.png)')

    parser.add_argument('--flow-input-posefile', default='pose_lcam_front.txt',
                        help='input pose file for flow generation (default: pose_lcam_front.txt)')

    # for imu generation
    parser.add_argument('--imu-outdir', default='imu',
                        help='output IMU files to this folder (default: imu)')

    parser.add_argument("--image-fps", type=int, default=10, 
                        help="image frame rate used for IMU generation")

    parser.add_argument("--imu-fps", type=int, default=100, 
                        help="IMU frame rate used for IMU generation")

    parser.add_argument('--imu-input-posefile', default='pose_lcam_front.txt',
                        help='posefile used for imu generation')

    # fisheye generation    
    parser.add_argument('--modalities', type=str, default="image,depth,seg",
            help='Comma-separated list of input modalities to postprocess. (default: "image,depth,seg")')
    
    parser.add_argument('--new-cam-models', type=str, default="fish,equirect",
        help='Comma-separated list of camera models to generate. (default: "fish,equirect")')
    
    parser.add_argument('--fish-video-only', action='store_true', default=False,
                        help='generate preview video with existing data(default: False)')
    
    # zip 
    parser.add_argument('--zip-outdir', default='',
                        help='output zip files to this folder')

    parser.add_argument('--upload-targets', default='azure,cluster',
                        help='upload platforms, split by comma')

    args = parser.parse_args()

    return args
