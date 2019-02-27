from .networks_base import TrackingNetworkBase
from .blocks import *
from .helpers import *


class TrackingNetwork(TrackingNetworkBase):

    def __init__(self, training=False, batch_size=1):
        TrackingNetworkBase.__init__(self)
        self._batch_size = batch_size
        self._placeholders = {
            'depth_key': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1)),
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3)),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 3)),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4), name='intrinsics'),
            'prev_rotation': tf.placeholder(tf.float32, shape=(batch_size, 3), name='prev_rotation'),
            'prev_translation': tf.placeholder(tf.float32, shape=(batch_size, 3), name='prev_translation'),
        }
        if training:
            self._placeholders['gt_rotation'] = tf.placeholder(tf.float32, shape=(self._batch_size, 3), name='gt_rotation')
            self._placeholders['gt_translation'] = tf.placeholder(tf.float32, shape=(self._batch_size, 3), name='gt_translation')

    def build_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation):
        _weights_regularizer = None
        depth_normalized0 = convert_NHWC_to_NCHW(depth_key)
        key_image0 = convert_NHWC_to_NCHW(image_key)
        current_image0 = convert_NHWC_to_NCHW(image_current)
        batch_size = depth_key.get_shape().as_list()[0]

        # depth_normalized0 = depth_key
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
            print('m1')
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
            print('m2')
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
            print('m3')
            r_abs, t_abs = apply_motion_increment(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1),
                                                  tf.expand_dims(motion_prediction_list[-1]['predict_translation'], axis=1),
                                                  motion_inc_prediction['predict_rotation'],
                                                  motion_inc_prediction['predict_translation'], )
            motion_prediction_abs = {
                'predict_rotation': r_abs,
                'predict_translation': t_abs,
            }
            motion_prediction_list.append(motion_prediction_abs)


        print('last')
        num_samples = motion_inc_prediction['num_samples']
        rotation_samples = tf.transpose(motion_inc_prediction['predict_rotation_samples'], [0, 2, 1])
        translation_samples = tf.transpose(motion_inc_prediction['predict_translation_samples'], [0, 2, 1])
        prev_rotation_tiled = tf.tile(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1), [1, num_samples, 1])
        prev_translation_tiled = tf.tile(tf.expand_dims(motion_prediction_list[-1]['predict_rotation'], axis=1), [1, num_samples, 1])
        rot_samples_abs, transl_samples_abs = apply_motion_increment(prev_rotation_tiled,
                                                                     prev_translation_tiled,
                                                                     rotation_samples,
                                                                     translation_samples, )

        result = {}

        motion_samples_abs = tf.concat((rot_samples_abs, transl_samples_abs), axis=2)
        motion_abs = tf.concat((r_abs, t_abs), axis=1, name='result')
        result['motion_samples_abs'] = motion_samples_abs
        result['motion_abs'] = motion_abs

        deviations = motion_samples_abs - tf.expand_dims(motion_abs, axis=1)  # [batch_size,num_predictions,6]
        sigma = tf.matmul(deviations, deviations, transpose_a=True) / num_samples
        epsilon = 0.1
        sigma = tf.add(sigma, epsilon * tf.eye(6, 6, batch_shape=[sigma.shape[0]], dtype=sigma.dtype), name='covariance')

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

    def build_training_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation, gt_rotation, gt_translation, learning_rate=0.1):
        result = self.build_net(depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation)

        # ground truth
        gt_x = tf.concat([gt_rotation, gt_translation], 1, name='gt_x')

        # motion loss
        alpha = 0.5
        r_norm = tf.norm(result['predict_rotation'] - gt_rotation, axis=1)
        t_norm = tf.norm(result['predict_translation'] - gt_translation, axis=1)
        motion_loss = tf.reduce_mean(tf.add(alpha*r_norm, t_norm), axis=0, name='motion_loss')
        tf.summary.scalar('motion loss', motion_loss)

        # flow loss
        flow_loss = 0
        # tf.summary.scalar('flow loss', flow_loss)

        # uncertainty loss
        x = tf.subtract(result['motion_abs'], gt_x, name='x')
        m = tf.matmul(tf.expand_dims(x, 1), result['covariance'])
        m = tf.matmul(m, tf.expand_dims(x, 1), transpose_b=True)
        m = tf.squeeze(m)
        # todo bessel function
        uncertainty_loss = tf.reduce_mean(0.5*tf.log(tf.norm(result['covariance'], axis=[-2, -1])) - 2*tf.log(m/2) - tf.log(tf.sqrt(2*m)), axis=0)
        tf.summary.scalar('uncertainty loss', uncertainty_loss)

        # overall loss
        tracking_loss = motion_loss + flow_loss + uncertainty_loss
        tf.summary.scalar('tracking_loss', tracking_loss)
        result['loss'] = tracking_loss
        result['motion_loss'] = motion_loss
        result['uncertainty_loss'] = uncertainty_loss
        result['summary'] = tf.summary.merge_all()
        return result
