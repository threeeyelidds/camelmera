# tar and compress the files according to different modality folders, 
# all the trajectories in one env will be pack together
# Upload the data to Azure/Cluster/PSC
# Upload the folders or upload the zip files
import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

from settings import get_args

from os import system, mkdir
from os.path import join, split, isdir, isfile
from data_enumeration import enumerate_trajs

TARTANV2_DIR = "https://tartanairv2.blob.core.windows.net"
TOCKEN = "?sv=2021-04-10&st=2023-03-24T13%3A54%3A41Z&se=2023-04-30T13%3A54%3A00Z&sr=c&sp=racwdxltf&sig=ysAbUStRig2DjKKA11cnivjQhGp2INo7O17LJkfYpuw%3D"

def run_cmd(cmd):
    print("===>",cmd)
    system(cmd)

class UploaderBase(object):
    def __init__(self,):
        self.platform = "Base"

    def upload_folder(self, source_folder, target_folder):
        '''
        the source_folder is a full directory on the local machine
        the source_folder will be uploaded as a sub-folder in the target_folder
        '''
        raise NotImplementedError

    def upload_file(self, source_file, target_folder):
        '''
        the source_file is a full path of a file on the local machine
        the source_file will be uploaded to the target_folder on the remote machine
        '''
        raise NotImplementedError

    def upload_files(self, source_folder, source_file_list, target_folder): 
        '''
        the source_folder is a path on the local machine
        the source_file_list is a list of file names under source_folder
        the files will be uploaded to the target_folder without a sub-folder on the remote machine
        '''
        raise NotImplementedError

class AzureUploader(UploaderBase):
    def __init__(self,):
        self.platform = "Azure"

    def upload_folder(self, source_folder, target_folder):
        '''
        target_folder example: /data-raw/ENV_NAME/Data_easy
        '''
        cmd = 'azcopy copy ' + source_folder  + ' "' + TARTANV2_DIR + target_folder + TOCKEN + '" --recursive --as-subdir=true' 
        run_cmd(cmd)

    def upload_file(self, source_file, target_file):
        cmd = 'azcopy copy "' + source_file + '" "' + TARTANV2_DIR + target_file + TOCKEN + '" ' 
        run_cmd(cmd)

    def upload_files(self, source_folder, source_file_list, target_folder):
        '''
        TODO: to be tested
        '''
        fileliststr = ';'.join(source_file_list)
        cmd = 'azcopy copy ' + source_folder + ' "' + TARTANV2_DIR + target_folder + TOCKEN + '" --include-path "' + fileliststr + '" --as-subdir=false' 
        run_cmd(cmd)

class ClusterUploader(UploaderBase):
    def __init__(self,):
        self.platform = "Cluster"

    def upload_folder(self, source_folder, target_folder):
        cmd = 'scp -r ' + source_folder + ' ' + 'wenshanw@perceptron.ri.cmu.edu:' + target_folder
        run_cmd(cmd)

    def upload_file(self, source_file, target_folder):
        cmd = 'scp ' + source_file + ' ' + 'wenshanw@perceptron.ri.cmu.edu:' + target_folder
        run_cmd(cmd)

    def upload_files(self, source_folder, source_file_list, target_folder):
        sourcelist = [join(source_folder, ff) for ff in source_file_list]
        fileliststr = ' '.join(sourcelist)
        cmd = 'scp ' + fileliststr + ' ' + 'wenshanw@perceptron.ri.cmu.edu:' + target_folder 
        run_cmd(cmd)

class PSCUploader(UploaderBase):
    def __init__(self,):
        self.platform = "PSC"

    def upload_folder(self, source_folder, target_folder):
        cmd = 'scp -r ' + source_folder + ' ' + 'wenshanw@bridges2.psc.edu:' + target_folder
        run_cmd(cmd)

    def upload_file(self, source_file, target_folder):
        cmd = 'scp ' + source_file + ' ' + 'wenshanw@bridges2.psc.edu:' + target_folder
        run_cmd(cmd)
    
    def upload_files(self, source_folder, source_file_list, target_folder):
        sourcelist = [join(source_folder, ff) for ff in source_file_list]
        fileliststr = ' '.join(sourcelist)
        cmd = 'scp ' + fileliststr + ' ' + 'wenshanw@bridges2.psc.edu:' + target_folder 
        run_cmd(cmd)

def zip_file(source_folder, source_file_list, target_file):
    '''
    the source_folder will be compressed into gz file, and stored as target_file
    '''
    fileliststr = ' '.join(source_file_list)
    cmd = 'cd ' + source_folder + ';tar -czvf ' + target_file + ' ' + fileliststr
    run_cmd(cmd)

# def upload_folders(uploader, sourcefolderlist, targetfolderlist):
#     pass

# def upload_files(uploader, sourcefilelist, targetfilelist):
#     pass

# def gz_folders(sourcefolderlist, targetfilelist):
#     pass

# depth, image, seg
# lcam-fblrub, rcam-fblrub, fish, equirect
# imu, lidar, flow
# 51 folders
folderlist = [
    "depth_lcam_back", 
    "depth_lcam_bottom", 
    "depth_lcam_equirect", 
    "depth_lcam_fish", 
    "depth_lcam_front", 
    "depth_lcam_left", 
    "depth_lcam_right", 
    "depth_lcam_top", 
    "depth_rcam_back", 
    "depth_rcam_bottom", 
    "depth_rcam_equirect", 
    "depth_rcam_fish", 
    "depth_rcam_front", 
    "depth_rcam_left", 
    "depth_rcam_right", 
    "depth_rcam_top", 
    "flow_lcam_front", 
    "image_lcam_back", 
    "image_lcam_bottom", 
    "image_lcam_equirect", 
    "image_lcam_fish", 
    "image_lcam_front", 
    "image_lcam_left", 
    "image_lcam_right", 
    "image_lcam_top", 
    "image_rcam_back", 
    "image_rcam_bottom", 
    "image_rcam_equirect", 
    "image_rcam_fish", 
    "image_rcam_front", 
    "image_rcam_left", 
    "image_rcam_right", 
    "image_rcam_top", 
    "imu", 
    "lidar", 
    "seg_lcam_back", 
    "seg_lcam_bottom", 
    "seg_lcam_equirect", 
    "seg_lcam_fish", 
    "seg_lcam_front", 
    "seg_lcam_left", 
    "seg_lcam_right", 
    "seg_lcam_top", 
    "seg_rcam_back", 
    "seg_rcam_bottom", 
    "seg_rcam_equirect", 
    "seg_rcam_fish", 
    "seg_rcam_front", 
    "seg_rcam_left", 
    "seg_rcam_right", 
    "seg_rcam_top", 
]

# 21 folders
pubfolderlist = [
    "depth_lcam_equirect", 
    "depth_lcam_fish", 
    "depth_lcam_front", 
    "depth_rcam_equirect", 
    "depth_rcam_fish", 
    "depth_rcam_front", 
    "flow_lcam_front", 
    "image_lcam_equirect", 
    "image_lcam_fish", 
    "image_lcam_front", 
    "image_rcam_equirect", 
    "image_rcam_fish", 
    "image_rcam_front", 
    "imu", 
    "lidar", 
    "seg_lcam_equirect", 
    "seg_lcam_fish", 
    "seg_lcam_front", 
    "seg_rcam_equirect", 
    "seg_rcam_fish", 
    "seg_rcam_front", 
]

def get_all_package_list(data_root_dir, envlist = None, datatypes = ['Data_easy', 'Data_hard']):
    '''
    Return a list of dictionary
    Each dictionary corresponds to a final zip file
    The key of the dict is the zip/gz file name
    The value of the dict is a list of files in the zip/gz file 
    Example: 
    [
        "abandonedfactory/Data_easy/depth_lcam_equirect": ["abandonedfactory/Data_easy/P000/depth_lcam_equirect","abandonedfactory/Data_easy/P001/depth_lcam_equirect",...]
        "abandonedfactory/Data_easy/flow_lcam_front": ["abandonedfactory/Data_easy/P000/flow_lcam_front","abandonedfactory/Data_easy/P001/flow_lcam_front",...]
        ...
    ]
    '''
    trajdict = {}
    for datatype in datatypes:
        trajdict[datatype] = enumerate_trajs(data_root_dir, data_folders = [datatype])

    if envlist is None: 
        envs = trajdict[datatypes[0]].keys()
    else:
        envs = envlist
    zipfiledict = {}
    for env in envs:
        envdir = join(data_root_dir, env)
        for motiontype in datatypes:
            typedir = join(envdir, motiontype)
            trajlist = trajdict[motiontype][env] # Data_easy/P000
            for folder in pubfolderlist: 
                outfilename = join(env, motiontype, folder)
                outfilelist = []
                for traj in trajlist:
                    outfilelist.append(join(env, traj, folder))
                outfilelist.append(join(env, traj, 'pose_lcam_front.txt'))
                outfilelist.append(join(env, traj, 'pose_rcam_front.txt'))
                zipfiledict[outfilename] = outfilelist
    return zipfiledict
            
def generate_zip_files(data_root_dir, zip_out_dir, envlist, datatypes):
    '''
    data_root_dir: the root folder containing all the environments
    zip_out_dir: the root folder for the output compressed files
    '''
    zipfiledict = get_all_package_list(data_root_dir, envlist, datatypes)
    output_ziplist = []
    # import ipdb;ipdb.set_trace()
    for (zipfilename, filelist) in zipfiledict.items():
        zipfiledir, foldername = split(zipfilename)
        envname, datatype = split(zipfiledir)

        envfolder = join(zip_out_dir, envname)
        datafolder = join(envfolder, datatype)

        if not isdir(envfolder):
            mkdir(envfolder)
        if not isdir(datafolder):
            mkdir(datafolder)

        target_file = join(datafolder, foldername + '.tar.gz')
        zip_file(data_root_dir, filelist, target_file)
        output_ziplist.append(target_file)

    return output_ziplist

def upload_raw_by_traj(uploader, data_root_dir, env_folders, data_folders, upload_trajs=True, upload_debugfiles=True):
    folderlist = ['analyze', 'video']
    filelist = ['seg_label.json']
    for data_folder in data_folders:
        filelist.append(join(data_folder, 'sample.log'))

    for env in env_folders:
        env_dir = join(data_root_dir, env)
        print("====> Uploading {}".format(env_dir))

        if upload_trajs:
            trajdict = enumerate_trajs(data_root_dir,  data_folders = data_folders)
            if env not in trajdict:
                print("!!! Error missing: {}".format(env_dir))
            trajlist = trajdict[env]
            for traj in trajlist: 
                trajdir = join(env_dir, traj)
                datastr, _ = split(traj)
                print("  *** Uploading {}".format(trajdir))
                uploader.upload_folder(trajdir, '/data-raw/'+env+'/'+datastr)

        if upload_debugfiles:
            # upload analyze and video folder
            for folder in folderlist:
                folderdir = join(env_dir, folder)
                if not isdir(folderdir):
                    print("!!! Error missing folder {}".format(folderdir))
                    continue
                print("   *** uploading {}".format(folderdir))
                uploader.upload_folder(folderdir, '/data-raw/'+env)

            for file in filelist:
                filedir = join(env_dir, file)
                if not isfile(filedir):
                    print("Error missing file {}".format(filedir))
                    continue
                print("   *** uploading {}".format(filedir))
                targetfile = '/data-raw/'+ env + '/' + file
                uploader.upload_file(filedir, targetfile)

# python upload_data.py 
# --data-root /home/amigo/tmp/test_root 
# --data-folders "Data_easy" 
# --env-folders "coalmine" 
# --zip-outdir /home/amigo/tmp/test_root_zip

# python upload_data.py --data-root E:\TartanAir_v2 --env-folders OldScandinaviaExposure --upload-targets azure
# C:\programs\tartanair\sample_pipeline\src\postprocessing  and run python upload_data.py --data-root E:\tartanair-v2\data --env-folders ConstructionSite,DesertGasStationExposure --upload-targets azure
if __name__=="__main__":
    args = get_args()

    data_root_dir = args.data_root
    data_folders = args.data_folders.split(',')
    env_folders  = args.env_folders.split(',')
    zip_out_dir = args.zip_outdir

    upload_targets = args.upload_targets.split(',')

    if 'azure' in upload_targets:
        uu = AzureUploader()
        upload_raw_by_traj(uu, data_root_dir, env_folders, data_folders)
            # trajdict = enumerate_trajs(data_root_dir)
            # if env not in trajdict:
            #     print("Error missing: {}".format(env_dir))
            # trajlist = trajdict[env]
            # for traj in trajlist: 
            #     trajdir = join(env_dir, traj)
            #     datastr, _ = split(traj)
            #     print("  *** Uploading {}".format(trajdir))
            #     uu.upload_folder(trajdir, 'data-raw', target_subfolder='/'+env+'/'+datastr)

    # if env_folders[0] == '': # env_folders not specified use all envs
    #     env_folders = None

    # ziplist = generate_zip_files(data_root_dir, zip_out_dir, env_folders, data_folders)

    # testfile = "C:\\tartanair-v2\\trajs\\3Dscan.zip"
    # testfolder = "C:\\tartanair-v2\\trajs\\AbandonedSchool\\"

    # testlistfolder = "C:\\tartanair-v2\\trajs"

    # # testfile = "/home/amigo/tmp/v2_trajectory/apocalyptic_city_pose.zip"
    # # testfolder = "/home/amigo/tmp/v2_trajectory/abandonedcable"

    # # testlistfolder = "/home/amigo/tmp/v2_trajectory"
    # testfilelist = ["abandoned_school_pose.zip", "abandonedcable_pose.zip", "abandonfactory2.zip"]
    
    # uu1 = AzureUploader()
    # uu2 = ClusterUploader()
    # uu3 = PSCUploader()

    # # uu1.upload_folder(testfolder, 'data-raw')
    # # uu1.upload_file(testfile, 'data-zip-public')
    # # uu1.upload_files(testlistfolder, testfilelist, 'data-zip-public')

    # target_folder = '/project/learningphysics/test'
    # uu2.upload_folder(testfolder, target_folder)
    # uu2.upload_file(testfile, target_folder)
    # uu2.upload_files(testlistfolder, testfilelist, target_folder)

    # # target_folder = '/ocean/projects/cis210086p/wenshanw/test'
    # # uu3.upload_folder(testfolder, target_folder)
    # # uu3.upload_file(testfile, target_folder)
    # # uu3.upload_files(testlistfolder, testfilelist, target_folder)

# python upload_data.py --data-root E:\tartanair-v2\data --env-folders GothicIslandExposure --upload-targets azure -> Need to upload all after easy/P006
# python upload_data.py --data-root E:\tartanair-v2\data --env-folders GothicIslandExposure --data-folders Data_easy --upload-targets azure 