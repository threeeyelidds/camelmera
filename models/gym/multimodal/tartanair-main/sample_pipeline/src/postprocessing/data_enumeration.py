from os.path import isfile, join, isdir
from os import listdir
import yaml

def enumerate_trajs(data_root_dir, data_folders = ['Data_easy','Data_hard']):
    '''
    Return a dict:
        res['env0']: ['Data_easy/P000', 'Data_easy/P001', ...], 
        res['env1']: ['Data_easy/P000', 'Data_easy/P001', ...], 
    '''
    env_folders = listdir(data_root_dir)    
    env_folders = [ee for ee in env_folders if isdir(join(data_root_dir, ee))]
    env_folders.sort()
    print('Detected envs {}'.format(env_folders))
    trajlist = {}
    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        # print('Working on env {}'.format(env_dir))
        trajlist[env_folder] = []
        for data_folder in data_folders:
            datapath = join(env_dir, data_folder)
            if not isdir(datapath):
                print('!!data folder missing '+ datapath)
                continue
            print('    Opened data folder {}'.format(datapath))

            trajfolders = listdir(datapath)
            trajfolders = [ join(data_folder, tf) for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()
            trajlist[env_folder].extend(trajfolders)
            print('    Found {} trajectories in env {}'.format(len(trajfolders), env_dir))
    return trajlist

def enumerate_modalities(trajfolder, filename = 'collection_config.yaml'):
    '''
    Read the config file, which is expected to be in the trajfolder
    According to the image_type, return a dictionary
    Return: 
        res['DepthPlanar']: a list of depth folder
        res['Scene']: a list of rgb folder
        res['Segmentation']: a list of seg folder
    '''
    config = yaml.safe_load(open(join(trajfolder, filename), 'r'))
    res = {} # 'DepthPlanner': [], 'Scene': [], 'Segmentation': []
    camera_list = config['camera_list']
    imagetype_list = config['image_type']
    camfolder_dict = config['camlist_name']
    typefolder_dict = config['type_name']
    for imgtype in imagetype_list:
        assert imgtype in typefolder_dict, "Unknown image type {}".format(imgtype)
        res[imgtype] = []
        for camstr in camera_list:
            assert camstr in camfolder_dict, "Unknown camera {}".format(camstr)
            res[imgtype].append(typefolder_dict[imgtype] + '_' + camfolder_dict[camstr])
    return res

def enumerate_frames(modfolder, surfix = '.png'):
    '''
    Return a list of frame index in the modfolder
    '''
    files = listdir(modfolder)
    files = [ff.split('_')[0] for ff in files if ff.endswith(surfix)]
    files.sort()
    return files

if __name__=="__main__":
    trajlist = enumerate_trajs('/home/amigo/tmp/test_root', 'Data_easy')
    envs = list(trajlist.keys())
    print(trajlist)
    # import ipdb;ipdb.set_trace()
    folderlist = enumerate_modalities('/home/amigo/tmp/test_root/' + envs[0]+'/'+trajlist[envs[0]][0])
    print(folderlist)