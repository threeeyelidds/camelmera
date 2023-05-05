# tartanair
Automatic data collection pipeline for the TartanAir dataset

## Preparation on Ubuntu 20.04

Install ROS Noetic on Ubuntu 20.04 [as usual](http://wiki.ros.org/noetic/Installation/Ubuntu).

Make sure you have `catkin tools` installed. Follow the instructions [here](https://catkin-tools.readthedocs.io/en/latest/installing.html).

Install more supporting packages.

```bash
sudo apt-get install -y ros-noetic-octomap-ros ros-noetic-dynamic-edt-3d 

pip install numpy scipy networkx pyyaml matplotlib pandas opencv-python
```

## Preparation of data collection on Windows

Apart from the above python packages, the msgpackrpc lib is not good by default, according to [this post]
(https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198). 

```bash
pip uninstall msgpack-python

pip install msgpack

cd C:\programs

git clone git@github.com:tbelhalfaoui/msgpack-rpc-python.git

cd msgpack-rpc-python

git checkout fix-msgpack-dep

python setup.py install --user
```

### Fix the unhashable bug of ROS Message object under Python3.
In ROS Neotic, the ROS message is not hashable, which ressults in a networkx error: 'TypeError: cannot unpack non-iterable Pose object\n'

```bash
sudo gedit /opt/ros/noetic/lib/python3/dist-packages/genpy/message.py
```

Add the following function to the `Message` class.

```python
def __hash__(self):
    return super().__hash__()
```

The bug is due to the behavior of Python3. It seems that in Python3, if a class has an overloaded `__eq__` function but does not have an explicit overload of the `__hash__` function, the instance of the class becomes unhashable.
