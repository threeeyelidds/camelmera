from __future__ import print_function

import copy
import numpy as np
import numpy.linalg as LA
import struct

ply_header_color = '''ply
format %(pt_format)s 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

ply_header = '''ply
format %(pt_format)s 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''
# pt_format
# binary_little_endian 1.0
# ascii 1.0

PLY_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#c0392b",\
    ]

PLY_COLOR_LEVELS = 20

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict

def color_map(data, colors, nLevels):
    # Get the color gradient dict.
    gradientDict = polylinear_gradient(colors, nLevels)

    # Get the actual levels generated.
    n = len( gradientDict["hex"] )

    # Level step.
    dMin = data.min()
    dMax = data.max()
    step = ( dMax - dMin ) / (n-1)

    stepIdx = ( data - dMin ) / step
    stepIdx = stepIdx.astype(np.int32)

    rArray = np.array( gradientDict["r"] )
    gArray = np.array( gradientDict["g"] )
    bArray = np.array( gradientDict["b"] )

    r = rArray[ stepIdx ]
    g = gArray[ stepIdx ]
    b = bArray[ stepIdx ]

    return r, g, b

def write_ply(fn, verts, colors=None, pt_format='binary_little_endian'):
    '''
    pt_format: text: ascii
               binary: 'binary_little_endian'
    '''
    verts  = verts.reshape(-1, 3)
    if colors is None:
        fmtstr = '%f %f %f'
        headerstr = (ply_header % dict(vert_num=len(verts), pt_format=pt_format)).encode('utf-8')
    else:
        fmtstr = '%f %f %f %d %d %d'
        headerstr = (ply_header_color % dict(vert_num=len(verts), pt_format=pt_format)).encode('utf-8')
        colors = colors.reshape(-1, 3)
        verts  = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write(headerstr)
        # np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        if pt_format == 'ascii':
            np.savetxt(f, verts, fmt=fmtstr)
        elif pt_format == 'binary_little_endian':
            for i in range(verts.shape[0]):
                if colors is None:
                    f.write(bytearray(struct.pack("fff",verts[i,0],verts[i,1],verts[i,2])))
                else:
                    f.write(bytearray(struct.pack("fffccc",verts[i,0],verts[i,1],verts[i,2],
                                colors[i,0].tobytes(), colors[i,1].tobytes(), colors[i,2].tobytes())))
        else:
            print("ERROR: Unknow PLY format {}".format(pt_format))

def output_to_ply(fn, X, imageSize, rLimit, origin, format, use_rgb=False):
    # Check the input X.
    if ( X.max() <= X.min() ):
        raise Exception("X.max() = %f, X.min() = %f." % ( X.max(), X.min() ) )
    
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = np.float32)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))
    rv = copy.deepcopy(vertices)
    rv[:, 0] = vertices[:, 0] - origin[0, 0]
    rv[:, 1] = vertices[:, 1] - origin[1, 0]
    rv[:, 2] = vertices[:, 2] - origin[2, 0]

    r = LA.norm(rv, axis=1).reshape((-1,1))
    mask = r < rLimit
    mask = mask.reshape(( mask.size ))
    # import ipdb; ipdb.set_trace()
    r = r[ mask ]

    if use_rgb:

        cr, cg, cb = color_map(r, PLY_COLORS, PLY_COLOR_LEVELS)

        colors = np.zeros( (r.size, 3), dtype = np.uint8 )

        colors[:, 0] = cr.reshape( cr.size )
        colors[:, 1] = cg.reshape( cr.size )
        colors[:, 2] = cb.reshape( cr.size )
    else:
        colors = None

    write_ply(fn, vertices[mask, :], colors, format)