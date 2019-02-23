from .networks_base import TrackingNetworkBase
from .blocks import *
from .helpers import *


class TrackingNetwork(TrackingNetworkBase):

    def __init__(self, batch_size=1):
        TrackingNetworkBase.__init__(self, batch_size)
        self._batch_size = batch_size
        self._placeholders = {
            'depth_key': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1)),
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3)),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3)),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4)),
            'prev_rotation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'prev_translation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
        }

    def build_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation):
        _weights_regularizer = None
        depth_key = tf.transpose(depth_key, [0, 3, 1, 2])
        image_key = tf.transpose(image_key, [0, 3, 1, 2])
        image_current = tf.transpose(image_current, [0, 3, 1, 2])
        batch_size = depth_key.get_shape().as_list()[0]
        depth_normalized = depth_key

        depth_normalized0 = depth_normalized
        depth_normalized1 = scale_tensor(depth_normalized0, -1)
        depth_normalized2 = scale_tensor(depth_normalized1, -1)

        key_image0 = image_key
        key_image1 = scale_tensor(key_image0, -1)
        key_image2 = scale_tensor(key_image1, -1)

        current_image0 = image_current
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
            r_abs, t_abs = apply_motion_increment(motion_prediction_list[-1]['predict_rotation'],
                                                  motion_prediction_list[-1]['predict_translation'],
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
            r_abs, t_abs = apply_motion_increment(motion_prediction_list[-1]['predict_rotation'],
                                                  motion_prediction_list[-1]['predict_translation'],
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
            r_abs, t_abs = apply_motion_increment(motion_prediction_list[-1]['predict_rotation'],
                                                  motion_prediction_list[-1]['predict_translation'],
                                                  motion_inc_prediction['predict_rotation'],
                                                  motion_inc_prediction['predict_translation'], )
            motion_prediction_abs = {
                'predict_rotation': r_abs,
                'predict_translation': t_abs,
            }
            motion_prediction_list.append(motion_prediction_abs)

        num_samples = motion_inc_prediction['num_samples']
        rotation_samples = tf.transpose(motion_inc_prediction['predict_rotation_samples'][0], [1, 0])
        translation_samples = tf.transpose(motion_inc_prediction['predict_translation_samples'][0], [1, 0])
        prev_rotation_tiled = tf.tile(motion_prediction_list[-1]['predict_rotation'], [num_samples, 1])
        prev_translation_tiled = tf.tile(motion_prediction_list[-1]['predict_translation'], [num_samples, 1])
        rot_samples_abs, transl_samples_abs = apply_motion_increment(prev_rotation_tiled,
                                                                     prev_translation_tiled,
                                                                     rotation_samples,
                                                                     translation_samples, )

        result = {}

        motion_samples_abs = tf.concat((rot_samples_abs, transl_samples_abs), axis=1)
        motion_abs = tf.concat((r_abs, t_abs), axis=1)
        result['motion_samples_abs'] = motion_samples_abs
        result['motion_abs'] = motion_abs

        deviations = tf.expand_dims(motion_samples_abs - motion_abs, axis=0)  # [1,num_predictions,6]
        sigma = tf.matmul(deviations, deviations, transpose_a=True) / num_samples
        epsilon = 0.1
        sigma = sigma + epsilon * tf.eye(6, 6, batch_shape=[batch_size], dtype=sigma.dtype)

        result.update(flow_inc_prediction)
        result.update(motion_prediction_abs)

        merged = tf.summary.merge_all()
        result['summary'] = merged

        result['rotation_samples'] = rot_samples_abs
        result['translation_samples'] = transl_samples_abs
        result['covariance'] = sigma

        # additional outputs for debugging
        result['warped_image'] = flow_inputs_and_gt['rendered_image_near']

        return result

    def build_training_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation, learning_rate=0.1):
        result = self.build_net(depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation)

        # ground truth
        gt_rotation = tf.placeholder(tf.float32, shape=(self._batch_size, 1, 3), name='gt_rotation')
        gt_translation = tf.placeholder(tf.float32, shape=(self._batch_size, 1, 3), name='gt_translation')
        gt_x = tf.concat([gt_rotation, gt_translation], 2, name='gt_x')

        # motion loss
        alpha = 0.5
        r_norm = tf.norm(result['predict_rotation'] - gt_rotation, axis=2)
        t_norm = tf.norm(result['predict_translation'] - gt_translation, axis=2)
        motion_loss = tf.add(alpha*r_norm, t_norm, name='motion_loss')
        tf.summary.scalar('motion loss', motion_loss)

        # flow loss
        flow_loss = 0
        tf.summary.scalar('flow loss', flow_loss)

        # uncertainty loss
        x = tf.reshape(tf.subtract(result['motion_abs'], gt_x), shape=(self._batch_size, 6, 1), name='x')
        tf.stop_gradient(x)
        diff = tf.reshape(result['motion_samples_abs'] - result['motion_abs'], shape=(self._batch_size, 64, 6, 1))
        sigma = tf.matmul(diff, diff, transpose_b=True)
        sigma = tf.reduce_mean(sigma, 1, name='sigma')
        m = tf.matmul(x, sigma, transpose_a=True)
        m = tf.matmul(m, x)
        m = tf.reshape(m, shape=(self._batch_size, 1))
        # todo bessel function
        uncertainty_loss = 0.5*tf.log(tf.norm(sigma, axis=[-2, -1])) - 2*tf.log(m/2) - tf.log(tf.sqrt(2*m))
        tf.summary.scalar('uncertainty loss', uncertainty_loss)

        # overall loss
        loss = motion_loss + flow_loss + uncertainty_loss
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return result, optimizer, loss, motion_loss, uncertainty_loss
