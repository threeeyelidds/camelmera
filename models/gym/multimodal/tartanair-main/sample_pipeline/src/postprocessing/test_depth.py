import numpy as np
import cv2
import os
from os.path import join
from os import listdir
import matplotlib.pyplot as plt

def vis_frame(trajdir, framestr, scale=0.5):
    rgbfolderlist = [
        'image_lcam_front',
        'image_lcam_back',
        'image_lcam_right',
        'image_lcam_left',
        'image_lcam_top',
        'image_lcam_bottom'
    ]
    # import ipdb;ipdb.set_trace()
    rgblist = []
    for w in [0,2,1,3]: # front, right, back, left
        rgbmodstr = rgbfolderlist[w].split('image_')[-1] # hard coded
        rgbfile_surfix = rgbmodstr + '.png'
        rgbfile = join(trajdir, rgbfolderlist[w], framestr + '_' + rgbfile_surfix)
        rgbnp = cv2.imread(rgbfile)
        rgbvis = cv2.resize(rgbnp, (0,0), fx=scale, fy=scale)
        rgblist.append(rgbvis)
    vis = np.concatenate(rgblist, axis=1)
    cv2.imshow('img', vis)
    cv2.waitKey(0)

anafolder = 'E:\\TartanAir_v2\\OldScandinaviaExposure\\analyze'
mod = 'Data_easy'
traj = "P004"
disp_max = np.load(join(anafolder, mod + '_'+ traj +'_disp_max.npy'))
disp_max_max = disp_max.max(axis=-1)
disp_max_mean = disp_max.mean(axis=-1)


print(disp_max_max.shape)
print(disp_max_mean.shape)

collision_mask = disp_max_max > 500
for k in range(len(disp_max)):
    trajdir = join(anafolder.split('analyze')[0], mod, traj)
    if collision_mask[k]:
        framestr = str(k).zfill(6)
        print(framestr, disp_max[k])
        vis_frame(trajdir, framestr)
# plt.plot(disp_max_max)
# plt.plot(disp_max_mean)
# plt.grid()

# plt.show()

# def mask_show_seg_id(segimg, seg_id):
#     seg_color = ind_to_color[seg_id]
#     vis = np.zeros_like(segimg, dtype=np.uint8)
#     vis[segimg == seg_color] = 255
#     cv2.imshow('img', vis)
#     # cv2.waitKey(0)

# def list_seg_id(segimg):
#     idlist = []
#     for id in range(1,255):
#         color = ind_to_color[id]
#         if np.sum(segimg == color) > 0:
#             # print("seg id {} - {}".format(id, color))
#             idlist.append(id)
#     return idlist

# import ipdb;ipdb.set_trace()
# find the object w largest seg id, visualize the seg and the rgb
# def read_seg_label(segfolder, targetid, skip=5, startfame=0):
#     segfiles = listdir(segfolder)
#     segfiles.sort()
#     for k in range(startfame, len(segfiles), skip):
#         segfile = segfiles[k]
#         segfilepath = join(segfolder, segfile)
#         segimg = cv2.imread(segfilepath, cv2.IMREAD_UNCHANGED)
#         idlist = list_seg_id(segimg)
#         if targetid in idlist:
#             print(segfile)
#             # find an image w/ the targetid
#             # visualize the seg and image
#             mask_show_seg_id(segimg, targetid)
#             # find the rgb image
#             rgbimgfile = join(segfolder.replace('seg', 'image'), segfile.replace('_seg', ''))
#             rgbimg = cv2.imread(rgbimgfile)
#             cv2.imshow('rgb', rgbimg)
#             cv2.waitKey(0)

# targetid =  list(segind_to_name.keys())[-1]
# print(targetid, segind_to_name[targetid])
# read_seg_label('E:\\TartanAir_v2\\ShoreCavesExposure\\Data_hard\\P000\\seg_lcam_back', targetid)

# segfile = "E:\\TartanAir_v2\\BrushifyMoonExposure\\Data_hard\\P005\\seg_lcam_back\\001141_lcam_back_seg.png"
# segfile = "E:\\TartanAir_v2\\BrushifyMoonExposure\\Data_easy\\P005\\seg_lcam_back\\000000_lcam_back_seg.png"
# segfile = "E:\\TartanAir_v2\\BrushifyMoonExposure\\Data_easy\\P000\\seg_lcam_back\\000000_lcam_back_seg.png"
# segfile = "C:\\tartanair-v2\\data\\test\\Data_easy\\P000\\seg_lcam_front\\000000_lcam_front_seg.png"
# segdir = "C:\\tartanair-v2\\data\\test\\Data_easy\\P001\\seg_lcam_left"
# segdir = "E:\\tartanair-v2\\data\\AbandonedCableExposure\\Data_easy\\P000\\seg_lcam_front"
# segfiles = listdir(segdir)
# segfiles.sort()
# segfiles = ["E:\\tartanair-v2\\data\\AbandonedCableExposure\\Data_hard\\P007\\seg_lcam_back\\000039_lcam_back_seg.png"]
# segfiles = ["E:\\tartanair-v2\\data\\AbandonedCableExposure\\Data_hard\\P007\\seg_lcam_front\\000273_lcam_front_seg.png"]
# segfiles = ["E:\\tartanair-v2\\data\\AmericanDinerExposure\\Data_hard\\P005\\seg_lcam_front\\000000_lcam_front_seg.png"]
# segfiles = ["C:\\tartanair-v2\\data\\test\\Data_hard\\P005\\seg_lcam_front\\000000_lcam_front_seg.png"]

# for k,segfile in enumerate(segfiles):
#     segfile = join(segdir, segfile)
#     segimg = cv2.imread(segfile, cv2.IMREAD_UNCHANGED)
#     list_seg_id(segimg)
#     print()
#     if k == 1:
#         break

#     mask_show_seg_id(segimg, ind_to_color[25])
#     mask_show_seg_id(segimg, ind_to_color[27])
#     mask_show_seg_id(segimg, ind_to_color[29])
#     mask_show_seg_id(segimg, ind_to_color[46])
#     mask_show_seg_id(segimg, ind_to_color[47])
#     mask_show_seg_id(segimg, ind_to_color[162])


# traj_list = [
#     'Data_easy_P000_seg.npy',
#     'Data_easy_P001_seg.npy',
#     'Data_easy_P002_seg.npy',
#     'Data_easy_P003_seg.npy',
#     'Data_easy_P004_seg.npy',
#     'Data_easy_P005_seg.npy',
#     'Data_hard_P000_seg.npy',
#     'Data_hard_P001_seg.npy',
#     'Data_hard_P002_seg.npy',
#     'Data_hard_P003_seg.npy',
#     'Data_hard_P004_seg.npy',
#     'Data_hard_P005_seg.npy'
# ]
# allpixelnums = {}

# for trajdir in traj_list:
#     pixelnums = np.load(join('E:\\TartanAir_v2\\BrushifyMoonExposure\\analyze_bk', trajdir), allow_pickle=True)
#     pixelnums = pixelnums.item()
#     for seglabel, pixelnum in pixelnums.items():
#         if seglabel not in allpixelnums:
#             allpixelnums[seglabel] = int(np.sum(pixelnum))
#         else:
#             allpixelnums[seglabel] += int(np.sum(pixelnum))
#         print(allpixelnums)
#         import ipdb;ipdb.set_trace()
