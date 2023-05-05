import os
import sys
# The path of the current Python script.
_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_PATH, '..'))

import json
from os.path import expanduser
from settings import get_args

'''
Usage: 
python AirsimSetting.py --data-collection --gamma 1.0 --min-exposure 0.1 --max-exposure 0.5
python AirsimSetting.py --mapping
'''
class AirsimSetting(object):
    def __init__(self):
        home = expanduser("~")
        self.settingfile = home+"/Documents/AirSim/settings.json"
        with open(self.settingfile,'r') as f:
            self.params = json.load(f)
        print(self.settingfile)
        print(self.params)

    def set_brightness(self, capturesetting, gamma, maxExposure=0.7, minExposure=0.3):
        cam0params = capturesetting[0] #self.params['CameraDefaults']['CaptureSettings'][0]
        cam0params['TargetGamma'] = gamma
        cam0params['AutoExposureMaxBrightness'] = maxExposure
        cam0params['AutoExposureMinBrightness'] = minExposure

    def set_resolution(self, capturesetting, width, height):
        # cam_params = self.params['CameraDefaults']['CaptureSettings']
        for camparam in capturesetting:
            camparam['Width'] = width
            camparam['Height'] = height

    def set_fov(self, capturesetting, fov):
        # cam_params = self.params['CameraDefaults']['CaptureSettings']
        for camparam in capturesetting:
            camparam['FOV_Degrees'] = fov

    def set_viewmode(self, display):
        if display:
            if self.params.has_key('ViewMode'):
                self.params['ViewMode'] = ''
        else:
            self.params['ViewMode'] = 'NoDisplay'

    def dumpfile(self):
        with open(self.settingfile, 'w') as f:
            json.dump(self.params, f, indent = 4, )
        print('setting.cfg file saved..')

    def enumerate_capturesetting(self, ):
        capturesettinglist = []
        if 'CameraDefaults' in self.params:
            capturesettinglist.append(self.params['CameraDefaults']['CaptureSettings'])
        if 'Vehicles' in self.params:
            if 'ComputerVision' in self.params['Vehicles']:
                if 'Cameras' in self.params['Vehicles']['ComputerVision']:
                    for cc in self.params['Vehicles']['ComputerVision']['Cameras']:
                        capturesettinglist.append(self.params['Vehicles']['ComputerVision']['Cameras'][cc]['CaptureSettings'])
        return capturesettinglist

    def set_gamma_all(self, gamma, maxExposure=0.7, minExposure=0.3):
        capturesettinglist = self.enumerate_capturesetting()
        for cs in capturesettinglist:
            self.set_brightness(cs, gamma, maxExposure, minExposure)

    def set_resolution_all(self, width, height):
        capturesettinglist = self.enumerate_capturesetting()
        for cs in capturesettinglist:
            self.set_resolution(cs, width, height)

    def set_fov_all(self, fov):
        capturesettinglist = self.enumerate_capturesetting()
        for cs in capturesettinglist:
            self.set_fov(cs, fov)

# deprecated
def set_mapping_setting():
    airsimSetting = AirsimSetting()
    airsimSetting.set_resolution(320, 320)
    airsimSetting.set_fov(90)
    airsimSetting.set_viewmode(display=False)
    airsimSetting.dumpfile()
    print('settings.json for mapping..')

def set_data_setting(gamma, maxExposure, minExposure):
    airsimSetting = AirsimSetting()
    airsimSetting.set_resolution_all(640, 640)
    airsimSetting.set_fov_all(90)
    # airsimSetting.set_viewmode(display=True)
    airsimSetting.set_gamma_all(gamma, maxExposure, minExposure)
    airsimSetting.dumpfile()
    print('settings.json for data collection, gamma {}, exposure ({}, {})'.format(gamma, maxExposure, minExposure))

if __name__ == '__main__':

    args = get_args()
    # if args.mapping:
    #     set_mapping_setting()
    # elif args.data_collection:
    set_data_setting(args.gamma, args.max_exposure, args.min_exposure)

    # airsimSetting = AirsimSetting()
    # airsimSetting.set_brightness(3.7)
    # airsimSetting.dumpfile()