from .base import Datatype
import cv2
from os.path import join
import numpy as np

class CubeDistanceType(Datatype):

    def save_file(self, data, fileindstr):
        """
        Save the data to hard drive.
        """
        data = np.expand_dims(data, axis=-1).view('<u1')
        cv2.imwrite(join(self.output_dir, fileindstr + '_' + self.cam_name+'_dist_cube.png'), data)
    