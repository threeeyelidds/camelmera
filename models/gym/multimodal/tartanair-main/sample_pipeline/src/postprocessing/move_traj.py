# # TODO: 
# 1. move and rename the traj
# 2. move and rename the video
# 3. move and rename the anafiles-traj
# 4. run the ana-update script to generate ana file for the env
import numpy as np
from os.path import isdir, isfile, join, split
from os import system, mkdir

def run_cmd(cmd):
    print("===>",cmd)
    system(cmd)

mvcmd = 'move '

analist = [
    'Data_easy_%s_segdata.npy',
    'Data_easy_%s_disp_hist.png',
    'Data_easy_%s_disp_max.npy',
    'Data_easy_%s_disp_mean.npy',
    'Data_easy_%s_disp_min.npy',
    'Data_easy_%s_disp_std.npy',
    'Data_easy_%s_rgb_max.npy',
    'Data_easy_%s_rgb_mean.npy',
    'Data_easy_%s_rgb_mean_std.png',
    'Data_easy_%s_rgb_min.npy',
    'Data_easy_%s_rgb_std.npy',
    'Data_easy_%s_seg.npy',
    'Data_easy_%s_seg_error.txt',

    'Data_hard_%s_segdata.npy',
    'Data_hard_%s_disp_hist.png',
    'Data_hard_%s_disp_max.npy',
    'Data_hard_%s_disp_mean.npy',
    'Data_hard_%s_disp_min.npy',
    'Data_hard_%s_disp_std.npy',
    'Data_hard_%s_rgb_max.npy',
    'Data_hard_%s_rgb_mean.npy',
    'Data_hard_%s_rgb_mean_std.png',
    'Data_hard_%s_rgb_min.npy',
    'Data_hard_%s_rgb_std.npy',
    'Data_hard_%s_seg.npy',
    'Data_hard_%s_seg_error.txt',
]

datafiles = [
    'data_%s_Data_easy_%s.txt', 
    'data_%s_Data_hard_%s.txt'
]

videofiles = [
    'Data_easy_%s_fish_equirect.mp4',
    'Data_easy_%s.mp4'
    'Data_hard_%s_fish_equirect.mp4',
    'Data_hard_%s.mp4'
]


def move_traj(source_traj, target_traj):
    '''
    source_traj: the absolute dir of the source traj
    target_traj: the absolute dir of the target traj
    '''
    # move and rename the traj
    assert isdir(source_traj), "Unknow traj: {}".format(source_traj)
    trajdir, traj = split(target_traj)
    assert isdir(trajdir), "unknow target folder: {}".format(trajdir)
    cmd = mvcmd + source_traj + ' ' + target_traj
    run_cmd(cmd)

def move_ana(source_ana, target_ana, source_traj, target_traj):
    '''
    source_ana: the source folder for the ana files
    target_ana: the target folder for the ana files
    source_traj: the trajectory folder name P00x
    target_traj: the trajectory folder name P00x
    '''
    # move 
    for ff in analist:
        sourcefile = join(source_ana, ff % (source_traj))
        targetfile = join(target_ana, ff % (target_traj))
        assert isfile(sourcefile), "Unknow video file: {}".format(sourcefile)
        filedir, filename = split(targetfile)
        assert isdir(filedir), "unknow target folder: {}".format(filedir)
        cmd = mvcmd + sourcefile + ' ' + targetfile
        run_cmd(cmd)


def move_video(source_vid, target_vid, source_traj, target_traj):
    for ff in videofiles:
        sourcefile = join(source_vid, ff % (source_traj))
        targetfile = join(target_vid, ff % (target_traj))
        assert isfile(sourcefile), "Unknow video file: {}".format(sourcefile)
        filedir, filename = split(targetfile)
        assert isdir(filedir), "unknow target folder: {}".format(filedir)

        cmd = mvcmd + sourcefile + ' ' + targetfile
        run_cmd(cmd)

# move_traj_files("D:\AbandonedSchoolExposure_bk", "E:\\testenv", "P001", "P005")
def move_traj_files(envfolder, target_envfolder, trajname, target_trajname ):
    '''
    Example: move_traj_files("E:\\AbandonedSchoolExposure", "E:\\testenv", "P000", "P005")
    this is the high level interface
    assume two trajectories Data_easy\P00X and Data_hard\P00X exsist
    envfolder: E:\AbandonSchool
    target_envfolder: E:\AbandonSchool
    trajname: P00X
    target_trajname: P00X
    '''
    move_traj(join(envfolder, "Data_easy", trajname), join(target_envfolder, "Data_easy", target_trajname))
    move_traj(join(envfolder, "Data_hard", trajname), join(target_envfolder, "Data_hard", target_trajname))

    move_ana(join(envfolder, 'analyze'), join(target_envfolder, 'analyze'), trajname, target_trajname)
    move_video(join(envfolder, 'video'), join(target_envfolder, 'video'), trajname, target_trajname)

def move_traj_to_collision_folder(envfolder, trajname):
    '''
    move the traj into the collision folder
    '''
    collision_folder = join(envfolder, "collision")
    if not isdir(collision_folder):
        mkdir(collision_folder)

    move_traj(join(envfolder, "Data_easy", trajname), join(collision_folder, "easy_" + trajname))
    move_traj(join(envfolder, "Data_hard", trajname), join(collision_folder, "hard_" + trajname))
    






foldercpcmd = 'xcopy /s /e '
filecpcmd = 'copy '

# used for copy data_easy or data_hard separately
analist_with_type = [
    '%s_segdata.npy',
    '%s_disp_hist.png',
    '%s_disp_max.npy',
    '%s_disp_mean.npy',
    '%s_disp_min.npy',
    '%s_disp_std.npy',
    '%s_rgb_max.npy',
    '%s_rgb_mean.npy',
    '%s_rgb_mean_std.png',
    '%s_rgb_min.npy',
    '%s_rgb_std.npy',
    '%s_seg.npy',
    '%s_seg_error.txt',
]

videofiles_with_type = [
    # '%s_fish_equirect.mp4',
    '%s.mp4'
]

def copy_traj(source_traj, target_traj):
    '''
    source_traj: the absolute dir of the source traj
    target_traj: the absolute dir of the target traj
    '''
    # move and rename the traj
    assert isdir(source_traj), "Unknow traj: {}".format(source_traj)
    trajdir, traj = split(target_traj)
    assert isdir(trajdir), "unknow target folder: {}".format(trajdir)
    target_traj = join(trajdir, traj, "") # add a \\ at the end
    cmd = foldercpcmd + source_traj + ' ' + target_traj
    run_cmd(cmd)

def copy_ana(source_ana, target_ana, source_traj, target_traj):
    '''
    source_ana: the source folder for the ana files
    target_ana: the target folder for the ana files
    source_traj: the trajectory folder name P00x
    target_traj: the trajectory folder name P00x
    '''
    typestr, trajstr = split(source_traj)
    source_str = typestr + "_" + trajstr
    for ff in analist_with_type:
        sourcefile = join(source_ana, ff % (source_str))
        targetfile = join(target_ana, ff % (target_traj))
        assert isfile(sourcefile), "Unknow ana file: {}".format(sourcefile)
        filedir, filename = split(targetfile)
        assert isdir(filedir), "unknow target folder: {}".format(filedir)
        cmd = filecpcmd + sourcefile + ' ' + targetfile
        run_cmd(cmd)

def copy_video(source_vid, target_vid, source_traj, target_traj):
    typestr, trajstr = split(source_traj)
    source_str = typestr + "_" + trajstr
    for ff in videofiles_with_type:
        sourcefile = join(source_vid, ff % (source_str))
        targetfile = join(target_vid, ff % (target_traj))
        assert isfile(sourcefile), "Unknow video file: {}".format(sourcefile)
        filedir, filename = split(targetfile)
        assert isdir(filedir), "unknow target folder: {}".format(filedir)

        cmd = filecpcmd + sourcefile + ' ' + targetfile
        run_cmd(cmd)


def copy_to_testing_set(envfolder, trajname, testingfolder, target_trajname):
    '''
    '''
    copy_traj(join(envfolder, trajname), join(testingfolder, target_trajname))

    if not isdir(join(testingfolder, 'analyze')):
        mkdir(join(testingfolder, 'analyze'))

    if not isdir(join(testingfolder, 'video')):
        mkdir(join(testingfolder, 'video'))

    copy_ana(join(envfolder, 'analyze'), join(testingfolder, 'analyze'), trajname, target_trajname)
    copy_video(join(envfolder, 'video'), join(testingfolder, 'video'), trajname, target_trajname)
    

if __name__=="__main__":
    # move_traj_files("D:\AbandonedSchoolExposure_bk", "E:\\testenv", "P001", "P005")
    copy_to_testing_set("F:\\HongKong", "Data_easy\\P004", "C:\\TartanAir_v2\\tartanair-v2-testing\\Data_easy", "P007")
    copy_to_testing_set("F:\\HongKong", "Data_hard\\P004", "C:\\TartanAir_v2\\tartanair-v2-testing\\Data_hard", "P007")    