"""
Generic template for what we need from each type for this class.
"""

class Datatype(object):

    def __init__(self, output_dir, cam_name):
        """
        Register the output folder for this type
        """
        self.output_dir = output_dir
        self.cam_name = cam_name

    def save_file(self, data, fileind):
        """
        Save the data to hard drive.
        """
        pass
