import numpy as np
import pykitti

# Change this to the directory where you store KITTI
basedir = '/home/loserleo/Downloads/homework'


def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.

    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset



import sys
sys.path.append("./source/")


import parseTrackletXML as xmlParser

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)
date = '2011_09_26'
drive = '0001'
dataset = load_dataset(date, drive)
tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '/home/loserleo/Downloads/homework/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
axes_limits = [
    [-20, 80],  # X axis range
    [-20, 20],  # Y axis range
    [-3, 10]  # Z axis range
]
axes_str = ['X', 'Y', 'Z']


def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.

    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)


def display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=0.2):


    dataset_velo = list(dataset.velo)


    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]

    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d != None:
            ax.set_xlim3d(xlim3d)
        if ylim3d != None:
            ax.set_ylim3d(ylim3d)
        if zlim3d != None:
            ax.set_zlim3d(zlim3d)

        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])

    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10, 30))
    plt.savefig('ground.png')
    plt.show()


    def draw_point_cloud_remove_ground(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Here is the code combie with the code last week,remove the ground point.

        """

        def remove_ground(dataset, step):

            # ground points grid filter
            def frange(start, stop, step):
                i = start
                while i < stop:
                    yield i
                    i += step
            dataset_Z_filtered = dataset[[0]]
            zfiltered = ((dataset[:,2].max() - dataset[:,2].min())/10) # setting height filtered from ground
            xstep = (dataset[:,0].max() - dataset[:,0].min()) / step
            ystep = (dataset[:,1].max() - dataset[:,1].min()) / step
            for x in frange(dataset[:, 0].min(), dataset[:, 0].max(), xstep):
                for y in frange(dataset[:, 1].min(), dataset[:, 1].max(), ystep):
                    datasetfiltered = dataset[(dataset[:, 0] > x)
                                              & (dataset[:, 0] < x + xstep)
                                              & (dataset[:, 1] > y)
                                              & (dataset[:, 1] < y + ystep)]
                    if datasetfiltered.shape[0] > 0:
                        datasetfiltered = datasetfiltered[datasetfiltered[:, 2]
                                                          > (datasetfiltered[:, 2].min() + zfiltered)]
                        if datasetfiltered.shape[0] > 0:
                            dataset_Z_filtered = np.concatenate((dataset_Z_filtered,
                                                                     datasetfiltered))
            return dataset_Z_filtered

        filter_data = remove_ground(velo_frame,2)

        print("Before filter, There are %d points. After filter, there are %d points \n" %(len(velo_frame),len(filter_data)))
        print("filter rate is %f" % (len(filter_data)/len(velo_frame)))

        ax.scatter(*np.transpose(filter_data[:, axes]), s=point_size, c=filter_data[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d != None:
            ax.set_xlim3d(xlim3d)
        if ylim3d != None:
            ax.set_ylim3d(ylim3d)
        if zlim3d != None:
            ax.set_zlim3d(zlim3d)
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])

    f3 = plt.figure(figsize=(15, 8))
    ax3 = f3.add_subplot(111, projection='3d')
    draw_point_cloud_remove_ground(ax3, 'Velodyne scan', xlim3d=(-10, 30))
    plt.savefig('noground.png')
    plt.show()


frame = 1

display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)






