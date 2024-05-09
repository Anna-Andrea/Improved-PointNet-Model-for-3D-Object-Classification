import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def se_block(input_feature, reduction_ratio=16):
    """Squeeze and Excitation Block"""
    num_channels = input_feature.get_shape()[-1]
    bottle_neck = int(num_channels // reduction_ratio)

    # Squeeze: Global average pooling across spatial dimensions
    squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

    # Excitation: Fully connected layers
    excitation = tf_util.fully_connected(squeeze, bottle_neck, activation_fn=tf.nn.relu, scope='fc1')
    excitation = tf_util.fully_connected(excitation, num_channels, activation_fn=tf.nn.sigmoid, scope='fc2')

    # Scale the input feature map
    scale = input_feature * excitation
    return scale

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    # 在 TensorFlow 2.x 中，不再使用 .value 来获取张量的值。相反，可以直接使用 Python 的 int() 函数将张量转换为整数值。
    # batch_size = point_cloud.get_shape()[0].value
    batch_size = int(point_cloud.get_shape()[0])
    # num_point = point_cloud.get_shape()[1].value
    num_point = int(point_cloud.get_shape()[1])
    end_points = {}

    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    
    # SE-Block
    with tf.compat.v1.variable_scope('se_block1') as sc:
        net = se_block(net, reduction_ratio=16)
        
    # Batch Normalization
    net = tf_util.batch_norm_for_conv2d(net, is_training, bn_decay, scope='bn1')

    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    
    # SE-Block
    with tf.compat.v1.variable_scope('se_block2') as sc:
        net = se_block(net, reduction_ratio=16)
        
    # Batch Normalization
    net = tf_util.batch_norm_for_conv2d(net, is_training, bn_decay, scope='bn2')
    
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    
    # Additional Convolutional Layers
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.compat.v1.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    # K = transform.get_shape()[1].value
    K = int(transform.get_shape()[1])
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.compat.v1.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
