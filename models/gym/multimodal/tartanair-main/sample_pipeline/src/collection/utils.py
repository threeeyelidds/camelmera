import numpy as np

def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

# mask = 1  : CROSS_OCC 
#      = 10 : SELF_OCC
#      = 100: OUT_OF_FOV
#      = 200: OVER_THRESHOLD
def flow32to16(flow32, mask8):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_16b = (flow_32b * 64) + 32768  
    '''
    # mask flow values that out of the threshold -512.0 ~ 511.984375
    mask1 = flow32 < -512.0
    mask2 = flow32 > 511.984375
    mask = mask1[:,:,0] + mask2[:,:,0] + mask1[:,:,1] + mask2[:,:,1]
    # convert 32bit to 16bit
    h, w, c = flow32.shape
    flow16 = np.zeros((h, w, 3), dtype=np.uint16)
    flow_temp = (flow32 * 64) + 32768
    flow_temp = np.clip(flow_temp, 0, 65535)
    flow_temp = np.round(flow_temp)
    flow16[:,:,:2] = flow_temp.astype(np.uint16)
    mask8[mask] = 200
    flow16[:,:,2] = mask8.astype(np.uint16)

    return flow16

def depth_rgba_float32(depth_rgba):
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def depth_float32_rgba(depth):
    '''
    depth: float32, h x w
    store depth in uint8 h x w x 4
    and use png compression
    '''
    depth_rgba = depth[...,np.newaxis].view("<u1")
    return depth_rgba
