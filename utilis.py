
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
import os
import json
import cv2
from scipy import ndimage

def read_rgb(rgb_file):
    rgb = io.imread(rgb_file)
    # plt.imshow(rgb)
    # plt.title(rgb_file)
    # plt.show()
    return rgb

def read_depth(depth_file):
    depth = io.imread(depth_file)
    # Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256)
    depth = depth * (1/ (256 * 256 * 256 - 1))
    # plt.imshow(depth)
    # plt.title(depth_file)
    # plt.show()
    return depth

def point_cloud_to_point_cloud(points,color ,K,transformation = None):
    points = np.transpose(points, (1,0))
    if transformation is not None:
        tmp = np.ones((4,points.shape[1]))
        tmp[:3,:] = points
        tmp = transformation @ tmp
    else:
        tmp = points
    tmp = K @ tmp
    tmp1 = tmp/tmp[2,:]
    return tmp1 

def depth_to_local_point_cloud(depth, color=None, k = np.eye(3),max_depth=1.0):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    "max_depth" is used to omit the points that are far enough.
    """
    "Reference: https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/carla/image_converter.py"
    far = 1000.0  # max depth in meters.
    normalized_depth = depth# depth_to_array(image)
    height, width = depth.shape

    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[width-1:-1:-1],
                     height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[height-1:-1:-1],
                     1, width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = np.delete(color, max_depth_indexes, axis=0)

    # pd2 = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = np.dot(np.linalg.inv(k), p2d)
    p3d *= normalized_depth * far
    
    p3d = np.transpose(p3d, (1,0))

    if color is not None:
        return p3d, color / 255.0
    else:
        return p3d, None
    
def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    
    Based code taken from:
    https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py    
    """

    classes = {
    0: [0, 0, 0], # None
    1: [70, 70, 70], # Buildings
    2: [190, 153, 153], # Fences
    3: [72, 0, 90], # Other
    4: [220, 20, 60], # Pedestrians
    5: [153, 153, 153], # Poles
    6: [157, 234, 50], # RoadLines
    7: [128, 64, 128], # Roads
    8: [244, 35, 232], # Sidewalks
    9: [107, 142, 35], # Vegetation
    10: [0, 0, 255], # Vehicles
    11: [102, 102, 156], # Walls
    12: [220, 220, 0] # TrafficSigns
    }
    array = image
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result.astype(np.uint8)


def predictionlabels_to_cityscapes_palette(prediction):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    
    Based code taken from:
    https://github.com/carla-simulator/data-collector/blob/master/carla/image_converter.py    
    """
    pred = np.argmax(prediction, axis=1)
    output=[]
    classes = {
    0: [0, 0, 0], # None
    1: [70, 70, 70], # Buildings
    2: [190, 153, 153], # Fences
    3: [72, 0, 90], # Other
    4: [220, 20, 60], # Pedestrians
    5: [153, 153, 153], # Poles
    6: [157, 234, 50], # RoadLines
    7: [128, 64, 128], # Roads
    8: [244, 35, 232], # Sidewalks
    9: [107, 142, 35], # Vegetation
    10: [0, 0, 255], # Vehicles
    11: [102, 102, 156], # Walls
    12: [220, 220, 0] # TrafficSigns
    }
    # array = image
    # result = np.zeros((array.shape[0], array.shape[1], 3))
    # for key, value in classes.items():
    #     result[np.where(array == key)] = value
    for i in pred:
        output.append(classes[i])
    output = np.asarray(output)/np.asarray(255.0)
    return output#result.astype(np.uint8)

print(predictionlabels_to_cityscapes_palette(np.array([[1,2,3],[2,6,4],[1.2,56.3,64]])))

def point_cloud_to_image(points,color ,K,transformation = None):
    points = np.transpose(points, (1,0))
    if transformation is not None:
        tmp = np.ones((4,points.shape[1]))
        tmp[:3,:] = points
        tmp = transformation @ tmp
    else:
        tmp = points
    tmp = K @ tmp
    tmp1 = tmp/tmp[2,:]
    # Note that multiple points might be mapped to the same pixel
    # The one with the lowest depth value should be assigned to that pixel
    # However, note this has not been implemented here
    # One may want to implement this
    u_cord = np.clip(np.round(tmp1[0,:]),0,511).astype(np.int)
    v_cord = np.clip(np.round(tmp1[1,:]),0,511).astype(np.int)
    if color is not None:
        imtmp = np.zeros((512,512,3)).astype(np.uint8)
        imtmp[u_cord, v_cord,:]= (color * 255).astype(np.uint8)
        
    else:
        imtmp = np.zeros((512,512)).astype(np.uint8)
        imtmp[u_cord, v_cord] = tmp[2,:]
        
    imtmp = cv2.flip(ndimage.rotate(imtmp, 90),1) # For some reason the axis were flipped
                                                  # therefore have been fixed here
        
    # plt.imshow(imtmp)
    # plt.show()
        
    return imtmp
    