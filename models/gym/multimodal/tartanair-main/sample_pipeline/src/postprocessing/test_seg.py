import numpy as np
import cv2
import os
from os.path import join
from os import listdir

from .data_verification import get_seg_color_label_value

labelfile = "E:\\TartanAir_v2\\ShoreCavesExposure\\seg_label.json"

color_to_ind, segname_to_ind, segind_to_name = get_seg_color_label_value(labelfile)
ind_to_color = {color_to_ind[k]: k for k in color_to_ind}


def mask_show_seg_id(segimg, seg_id):
    seg_color = ind_to_color[seg_id]
    vis = np.zeros_like(segimg, dtype=np.uint8)
    vis[segimg == seg_color] = 255
    cv2.imshow('img', vis)
    # cv2.waitKey(0)

def list_seg_id(segimg):
    idlist = []
    for id in range(1,255):
        color = ind_to_color[id]
        if np.sum(segimg == color) > 0:
            # print("seg id {} - {}".format(id, color))
            idlist.append(id)
    return idlist

# import ipdb;ipdb.set_trace()
# find the object w largest seg id, visualize the seg and the rgb
def read_seg_label(segfolder, targetid, skip=5, startfame=0):
    segfiles = listdir(segfolder)
    segfiles.sort()
    for k in range(startfame, len(segfiles), skip):
        segfile = segfiles[k]
        segfilepath = join(segfolder, segfile)
        segimg = cv2.imread(segfilepath, cv2.IMREAD_UNCHANGED)
        idlist = list_seg_id(segimg)
        if targetid in idlist:
            print(segfile)
            # find an image w/ the targetid
            # visualize the seg and image
            mask_show_seg_id(segimg, targetid)
            # find the rgb image
            rgbimgfile = join(segfolder.replace('seg', 'image'), segfile.replace('_seg', ''))
            rgbimg = cv2.imread(rgbimgfile)
            cv2.imshow('rgb', rgbimg)
            cv2.waitKey(0)

targetid =  list(segind_to_name.keys())[-1]
print(targetid, segind_to_name[targetid])
read_seg_label('E:\\TartanAir_v2\\ShoreCavesExposure\\Data_hard\\P000\\seg_lcam_back', targetid)

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
