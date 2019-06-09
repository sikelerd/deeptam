from .networks_base import TrackingNetworkBase
from .blocks import *
from .helpers import *
from scipy import special
from specialops.tf_specialops import points_to_depth
import tensorflow as tf

class TrackingNetwork(TrackingNetworkBase):

    def __init__(self, batch_size=1, training=False):
        TrackingNetworkBase.__init__(self)
        self._placeholders = {
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3), name='image_key'),
            'point_key': tf.placeholder(tf.float32, shape=(batch_size, None, 3), name='point_key'),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3), name='image_current'),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4), name='intrinsics'),
            'prev_rotation': tf.placeholder(tf.float32, shape=(batch_size, 3), name='prev_rotation'),
            'prev_translation': tf.placeholder(tf.float32, shape=(batch_size, 3), name='prev_translation'),
        }
        if training:
            self._placeholders['point_current'] = tf.placeholder(tf.float32, shape=(batch_size, None, 3), name='point_current')
            self._placeholders['gt_rotation'] = tf.placeholder(tf.float32, shape=(batch_size, 3), name='gt_rotation')
            self._placeholders['gt_translation'] = tf.placeholder(tf.float32, shape=(batch_size, 3), name='gt_translation')

    def build_net(self, image_key, point_key, image_current, intrinsics, prev_rotation, prev_translation):
        _weights_regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
        shape = image_key.get_shape().as_list()
        shape[3] = 1

        result = {}

        depth = points_to_depth(point_key)
        tf.stop_gradient(depth)
        result['depth'] = depth
        depth_normalized = tf.reshape(depth, (shape[0], 240, 320, 1))
        depth_normalized0 = convert_NHWC_to_NCHW(depth_normalized)
        key_image0 = convert_NHWC_to_NCHW(image_key)
        current_image0 = convert_NHWC_to_NCHW(image_current)

        depth_normalized1 = scale_tensor(depth_normalized0, -1)
        depth_normalized2 = scale_tensor(depth_normalized1, -1)

        # key_image0 = image_key
        key_image1 = scale_tensor(key_image0, -1)
        key_image2 = scale_tensor(key_image1, -1)

        # current_image0 = image_current
        current_image1 = scale_tensor(current_image0, -1)
        current_image2 = scale_tensor(current_image1, -1)

        motion_prediction_list = [{'predict_rotation': prev_rotation, 'predict_translation': prev_translation}]

        with tf.variable_scope("net_F1", reuse=None):
            flow_inputs_and_gt = create_flow_inputs_and_gt(
                key_image=key_image2,
                current_image=current_image2,
                intrinsics=intrinsics,
                depth=depth_normalized2,
                rotation=motion_prediction_list[-1]['predict_rotation'],
                translation=motion_prediction_list[-1]['predict_translation'],
                suffix='2',
            )
            flow_input = flow_inputs_and_gt['flow_input']
            flow_inc_prediction = flow_block(flow_input, weights_regularizer=_weights_regularizer)

        with tf.variable_scope("net_M1", reuse=None):
            motion_inputs = [
                (flow_inc_prediction['concat0'], 32),
                (tf.stop_gradient(flow_inputs_and_gt['rendered_depth_near_far']), 16),
            ]
            motion_inc_prediction1 = motion_block(motion_inputs, weights_regularizer=_weights_regularizer, resolution_level=2)
            motion_inc_prediction = motion_inc_prediction1
            r_abs, t_abs = apply_motion_increment(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1),
                                                  tf.expand_dims(motion_prediction_list[-1]['predict_translation'], axis=1),
                                                  motion_inc_prediction['predict_rotation'],
                                                  motion_inc_prediction['predict_translation'], )

            motion_prediction_abs = {
                'predict_rotation': r_abs,
                'predict_translation': t_abs,
            }
            motion_prediction_list.append(motion_prediction_abs)

        with tf.variable_scope("net_F2", reuse=None):
            flow_inputs_and_gt = create_flow_inputs_and_gt(
                key_image=key_image1,
                current_image=current_image1,
                intrinsics=intrinsics,
                depth=depth_normalized1,
                rotation=motion_prediction_list[-1]['predict_rotation'],
                translation=motion_prediction_list[-1]['predict_translation'],
                suffix='1',
            )

            flow_input = flow_inputs_and_gt['flow_input']
            flow_inc_prediction = flow_block(flow_input, weights_regularizer=_weights_regularizer)

        with tf.variable_scope("net_M2", reuse=None):
            motion_inputs = [
                (flow_inc_prediction['concat0'], 32),
                (tf.stop_gradient(flow_inputs_and_gt['rendered_depth_near_far']), 16),
            ]
            motion_inc_prediction2 = motion_block(motion_inputs, weights_regularizer=_weights_regularizer, resolution_level=1)
            motion_inc_prediction = motion_inc_prediction2
            r_abs, t_abs = apply_motion_increment(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1),
                                                  tf.expand_dims(motion_prediction_list[-1]['predict_translation'], axis=1),
                                                  motion_inc_prediction['predict_rotation'],
                                                  motion_inc_prediction['predict_translation'], )

            motion_prediction_abs = {
                'predict_rotation': r_abs,
                'predict_translation': t_abs,
            }
            motion_prediction_list.append(motion_prediction_abs)

        with tf.variable_scope("net_F3", reuse=None):
            flow_inputs_and_gt = create_flow_inputs_and_gt(
                key_image=key_image0,
                current_image=current_image0,
                intrinsics=intrinsics,
                depth=depth_normalized0,
                rotation=motion_prediction_list[-1]['predict_rotation'],
                translation=motion_prediction_list[-1]['predict_translation'],
                suffix='0',
            )
            flow_input = flow_inputs_and_gt['flow_input']
            flow_inc_prediction = flow_block(flow_input, weights_regularizer=_weights_regularizer)

        with tf.variable_scope("net_M3", reuse=None):
            motion_inputs = [
                (flow_inc_prediction['concat0'], 32),
                (tf.stop_gradient(flow_inputs_and_gt['rendered_depth_near_far']), 16),
            ]
            motion_inc_prediction3 = motion_block(motion_inputs, weights_regularizer=_weights_regularizer, resolution_level=0)
            motion_inc_prediction = motion_inc_prediction3
            r_abs, t_abs = apply_motion_increment(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1),
                                                  tf.expand_dims(motion_prediction_list[-1]['predict_translation'], axis=1),
                                                  motion_inc_prediction['predict_rotation'],
                                                  motion_inc_prediction['predict_translation'], )

            motion_prediction_abs = {
                'predict_rotation': r_abs,
                'predict_translation': t_abs,
            }
            motion_prediction_list.append(motion_prediction_abs)

        num_samples = motion_inc_prediction['num_samples']
        rotation_samples = tf.transpose(motion_inc_prediction['predict_rotation_samples'], [0, 2, 1])
        translation_samples = tf.transpose(motion_inc_prediction['predict_translation_samples'], [0, 2, 1])
        prev_rotation_tiled = tf.tile(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1), [1, num_samples, 1])
        prev_translation_tiled = tf.tile(tf.expand_dims(motion_prediction_list[-1]['predict_translation'], axis=1), [1, num_samples, 1])
        rot_samples_abs, transl_samples_abs = apply_motion_increment(prev_rotation_tiled,
                                                                     prev_translation_tiled,
                                                                     rotation_samples,
                                                                     translation_samples, )

        result.update(flow_inc_prediction)
        result.update(motion_prediction_abs)

        merged = tf.summary.merge_all()
        result['summary'] = merged

        result['rotation_samples'] = rot_samples_abs
        result['translation_samples'] = transl_samples_abs

        # additional outputs for debugging
        result['warped_image'] = flow_inputs_and_gt['rendered_image_near']

        return result

    def build_training_net(self, image_key, point_key, image_current, point_current, intrinsics, prev_rotation, prev_translation, gt_rotation, gt_translation):
        result = self.build_net(image_key, point_key, image_current, intrinsics, prev_rotation, prev_translation)

        with tf.variable_scope('loss'):
            # ground truth
            gt_x = tf.concat([gt_rotation, gt_translation], 1, name='gt_x')

            # motion loss
            alpha = 5
            r_norm = tf.norm(result['predict_rotation'] - gt_rotation, axis=1)
            t_norm = tf.norm(result['predict_translation'] - gt_translation, axis=1)
            motion_loss = tf.reduce_mean(tf.add(alpha*r_norm, t_norm), axis=0, name='motion_loss')
            # tf.summary.scalar('motion loss', motion_loss)

            # flow loss
            flow_loss = 0
            # tf.summary.scalar('flow loss', flow_loss)

            # uncertainty loss
            motion_samples_abs = tf.concat((result['rotation_samples'], result['translation_samples']), axis=2)
            motion_abs = tf.concat((result['predict_rotation'], result['predict_translation']), axis=1)
            deviations = motion_samples_abs - motion_abs
            sigma = tf.matmul(tf.expand_dims(deviations, -1), tf.expand_dims(deviations, -2))
            sigma = tf.reduce_mean(sigma, axis=1, name='covariance')

            def matrix_inverse(mat):
                # print('cond', np.linalg.cond(mat))
                if np.linalg.cond(mat) > 100:
                    mat = np.identity(mat.shape[-1])
                try:
                    return np.float32(np.linalg.inv(mat))
                except np.linalg.LinAlgError:
                    return np.float32(np.linalg.inv(mat + np.random.random_sample(mat.shape)))
            sigma_inv = tf.reshape(tf.py_func(matrix_inverse, [sigma], tf.float32), (-1, 6, 6))

            x = tf.stop_gradient(tf.subtract(motion_abs, gt_x, name='x'))
            m = tf.matmul(tf.expand_dims(x, -2), sigma_inv)
            m = tf.matmul(m, tf.expand_dims(x, -1))
            m = tf.squeeze(m, axis=[-1, -2])

            def modified_bessel(z):
                return np.float32(special.kv(4, z + 1e-4))

            uncertainty_loss = 0.5*tf.log(tf.norm(sigma, axis=[-2, -1]) + 1e-6) - 2*tf.log(m/2 + 1e-6) - tf.log(tf.py_func(modified_bessel, [tf.sqrt(2*m)], tf.float32) + 1e-6)
            uncertainty_loss = tf.abs(tf.reduce_mean(uncertainty_loss), name='uncertainty_loss')
            # tf.summary.scalar('uncertainty loss', uncertainty_loss)

            # distance loss
            # def apply_movement(rot, trans, points):
            #     new_points = []
            #     for i in range(rot.shape[0]):
            #         rot_m = angleaxis_to_rotation_matrix(rot[i])
            #         new_p = []
            #         for p in points[i]:
            #             new_p.append(rot_m.dot(p))
            #         new_p = np.array(new_p) - trans[i]
            #         new_points.append(new_p)
            #     new_points = np.array(new_points, dtype=np.float32)
            #     return new_points
            #
            # rotated_points = tf.py_func(apply_movement, [result['predict_rotation'], result['predict_translation'], point_key], tf.float32, stateful=False)
            # dists_key, _, _, _ = tf_nndistance.nn_distance(rotated_points, point_current)
            # distance_loss = tf.reduce_mean(dists_key)
            # tf.summary.scalar('distance_loss', distance_loss)

            # overall loss
            tracking_loss = (motion_loss + flow_loss + uncertainty_loss)
            tf.summary.scalar('tracking_loss', tracking_loss)
            result['loss'] = tracking_loss
            result['motion_loss'] = motion_loss
            result['uncertainty_loss'] = uncertainty_loss
            # result['distance_loss'] = distance_loss
        return result
