from .base import Datatype
import cv2
from os.path import join

class CubeSceneType(Datatype):

    def save_file(self, data, fileindstr):
        """
        Save the data to hard drive.
        """
        cv2.imwrite(join(self.output_dir, fileindstr + '_' + self.cam_name+'_cube.png'), data)
    