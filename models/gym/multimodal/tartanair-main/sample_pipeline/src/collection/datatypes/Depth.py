from .base import Datatype
import cv2
from os.path import join
from ..utils import depth_float32_rgba

class DepthType(Datatype):

    def save_file(self, data, fileindstr):
        """
        Save the data to hard drive.
        """
        depthrgba = depth_float32_rgba(data)
        cv2.imwrite(join(self.output_dir, fileindstr + '_' + self.cam_name+'_depth.png'), depthrgba)
    