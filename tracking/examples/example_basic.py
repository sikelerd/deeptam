from deeptam_tracker.tracker import TrackerCore
from deeptam_tracker.utils.vis_utils import convert_array_to_colorimg
from PIL import ImageChops
import matplotlib.pyplot as plt
import os
import numpy as np
from deeptam_tracker.utils.rotation_conversion import rotation_matrix_to_angleaxis
from deeptam_tracker.utils.datatypes import Pose
from minieigen import Matrix3, Vector3


def simple_visualization(image_key, image_cur, image_cur_virtual, frame_id):
    """Visualizes some image results
    
    image_key, image_cur, image_cur_virtual: np.array
    
    frame_id: int
    """
    image_key = convert_array_to_colorimg(image_key.squeeze())
    image_cur = convert_array_to_colorimg(image_cur.squeeze())
    image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze())

    diff = ImageChops.difference(image_cur, image_cur_virtual)  # difference should be small if the predicted pose is correct

    print('Close window to continue...')

    plt.subplot(2, 2, 1)
    plt.gca().set_title('Key frame image')
    fig = plt.imshow(image_key)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2, 2, 2)
    plt.gca().set_title('Current frame image {}'.format(frame_id))
    fig = plt.imshow(image_cur)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2, 2, 3)
    plt.gca().set_title('Virtual current frame image {}'.format(frame_id))
    fig = plt.imshow(image_cur_virtual)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2, 2, 4)
    plt.gca().set_title('Difference image')
    fig = plt.imshow(diff)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()


def main():
    # initialization
    examples_dir = os.path.dirname(__file__)
    checkpoint = os.path.join(examples_dir, '..', 'weights', 'deeptam_tracker_weights', 'snapshot-300000')
    datadir = os.path.join(examples_dir, '../../..', 'generating', '_out', 'Town01_0000.npy')
    # datadir = os.path.join(examples_dir, '../../..', 'generating', '_out', 'kitti', '04.npy')
    tracking_module_path = os.path.join(examples_dir, '..', 'python/deeptam_tracker/models/networks.py')

    sequence = np.load(datadir)
    WIDTH = 320
    HEIGHT = 240
    fw = WIDTH / (2 * np.tan(90 * np.pi / 360))
    fh = HEIGHT / (2 * np.tan(90 * np.pi / 360))
    cu = WIDTH / 2
    cv = HEIGHT / 2
    intrinsics = [[fw, fh, cu, cv]]

    # use first 3 frames as an example, the fisrt frame is selected as key frame
    frame_key = sequence[0]
    print(frame_key[2])
    print(Vector3(frame_key[2]))
    pose_key = Pose(R=Matrix3(frame_key[3]), t=Vector3(frame_key[2]))
    image_key = np.reshape(frame_key[0], (1, HEIGHT, WIDTH, 3))
    frame_1 = sequence[1]
    pose_1 = Pose(R=Matrix3(frame_1[3]), t=Vector3(frame_1[2]))
    image_1 = np.reshape(frame_1[0], (1, HEIGHT, WIDTH, 3))
    frame_2 = sequence[2]
    pose_2 = Pose(R=Matrix3(frame_2[3]), t=Vector3(frame_2[2]))
    image_2 = np.reshape(frame_2[0], (1, HEIGHT, WIDTH, 3))

    tracker_core = TrackerCore(tracking_module_path, checkpoint, intrinsics)

    # set the keyframe of tracker
    tracker_core.set_keyframe(image_key, np.full((1, 240, 320, 1), 0.01), pose_key)

    # track frame_1 w.r.t frame_key
    print('Track frame {} w.r.t key frame:'.format(1))
    results_1 = tracker_core.compute_current_pose(image_1, pose_key)
    pose_pr_1 = results_1['pose']

    print('prediction', rotation_matrix_to_angleaxis(pose_pr_1.R), pose_pr_1.t)
    print('gt', rotation_matrix_to_angleaxis(pose_1.R), pose_1.t)
    simple_visualization(image_key, image_1, results_1['warped_image'], 1)

    # track frame_2 w.r.t frame_key incrementally using pose_pr_1 as pose_guess
    print('Track frame {} w.r.t key frame incrementally using previous predicted pose:'.format(2))
    results_2 = tracker_core.compute_current_pose(image_2, pose_pr_1)
    pose_pr_2 = results_2['pose']

    print('prediction', rotation_matrix_to_angleaxis(pose_pr_2.R), pose_pr_2.t)
    print('gt', rotation_matrix_to_angleaxis(pose_2.R), pose_2.t)
    simple_visualization(image_key, image_2, results_2['warped_image'], 2)

    del tracker_core


if __name__ == "__main__":
    main()
