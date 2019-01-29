#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
    online segmentation using .npy & SqueezeSeg model

    this script can
                    1. read all .npy file from lidar_2d folder
                    2. predict label from SqueezeSeg model using tensorflow
                    3. publish to 'sqeeuze_seg/points' topic

    strongly inspried from [https://github.com/Durant35/SqueezeSeg]
    original code          [https://github.com/BichenWuUCB/SqueezeSeg]
"""


import sys
import os.path
import numpy as np
from PIL import Image
import struct
import tensorflow as tf

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

from squeezeseg.config import *
from squeezeseg.nets import SqueezeSeg
from squeezeseg.utils.util import *
from squeezeseg.utils.clock import Clock
from squeezeseg.imdb import kitti # ed: header added

fmt_full = ''



_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)
_NP_TYPES = {
    np.dtype('uint8')   :   (PointField.UINT8,  1),
    np.dtype('int8')    :   (PointField.INT8,   1),
    np.dtype('uint16')  :   (PointField.UINT16, 2),
    np.dtype('int16')   :   (PointField.INT16,  2),
    np.dtype('uint32')  :   (PointField.UINT32, 4),
    np.dtype('int32')   :   (PointField.INT32,  4),
    np.dtype('float32') :   (PointField.FLOAT32,4),
    np.dtype('float64') :   (PointField.FLOAT64,8)
}





class NPY_TENSORFLOW_TO_ROS():
    def __init__(self, pub_topic, FLAGS, npy_path="", npy_file_list=""):

        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
        self._mc = kitti_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        self._mc.BATCH_SIZE = 1         # TODO(bichen): fix this hard-coded batch size.
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        # ed: Publisher
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)


        rospy.Subscriber('/points_raw', PointCloud2, self.callback_pointcloud)


        rospy.spin()



    def _get_struct_fmt(self,cloud, field_names=None):
        fmt = '>' if cloud.is_bigendian else '<'
        offset = 0
        for field in (f for f in sorted(cloud.fields, key=lambda f: f.offset) if
                      field_names is None or f.name in field_names):
            if offset < field.offset:
                fmt += 'x' * (field.offset - offset)
                offset = field.offset
            if field.datatype not in _DATATYPES:
                print >> sys.stderr, 'Skipping unknown PointField datatype [%d]' % field.datatype
            else:
                datatype_fmt, datatype_length = _DATATYPES[field.datatype]
                fmt += field.count * datatype_fmt
                offset += field.count * datatype_length

        return fmt




    def pointcloud2_to_array(self,msg):
        global fmt_full
        if not fmt_full:
            fmt = self._get_struct_fmt(msg)
            fmt_full = '>' if msg.is_bigendian else '<' + fmt.strip('<>') * msg.width * msg.height
        # import pdb; pdb.set_trace()
        unpacker = struct.Struct(fmt_full)
        unpacked = np.asarray(unpacker.unpack_from(msg.data))
        # unpacked = unpacked.reshape(msg.height, msg.width, len(msg.fields))

        # Unpack RGB color info
        _float2rgb_vectorized = np.vectorize(_float2rgb)
        r, g, b = _float2rgb_vectorized(unpacked[:, :, 3])
        z = np.expand_dims(copy.deepcopy(unpacked[:, :, 2]), 2)
        r = np.expand_dims(r, 2)  # insert blank 3rd dimension (for concatenation)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)
        unpacked = np.concatenate((unpacked[:, :, 0:3], r, g, b), axis=2)

        return unpacked



    def callback_pointcloud(self, data):
        assert isinstance(data, PointCloud2)

        # record =self.pointcloud2_to_array(data)



        record = pc2.read_points(data)
        # record = np.asarray(record)
        record = np.asarray(list(record))
        # record = np.asarray(record)
        record=np.expand_dims(record,axis=0)


        # record=np.array(data)
        # record=np.array(data, dtype=np.float32)
        # record = np.asarray(data)

        lidar = record[:, :, :5]

        print 'record[:0] ' + str(record[:0])
        print 'record.shape ' + str(record.shape)


        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [1, lidar[:, :, 4].size , 1]
        )

        norm_lidar = (lidar - self._mc.INPUT_MEAN) / self._mc.INPUT_STD


        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [norm_lidar],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]

        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        cloud = np.stack((x, y, z, i, label))

        header = Header()
        header.stamp = rospy.Time().now()
        header.frame_id = "velodyne_link"

        # point cloud segments
        msg_segment = self.create_cloud_xyzil32(header, cloud.T)

        # publish
        self._pub.publish(msg_segment)
        rospy.loginfo("Point cloud processed. Took %.6f ms.", clock.takeRealTime())








    # Read all .npy data from lidar_2d folder
    def get_npy_from_lidar_2d(self, npy_path, npy_file_list):
        self.npy_path = npy_path
        self.npy_file_list = open(npy_file_list,'r').read().split('\n')
        self.npy_files = []

        for i in range(len(self.npy_file_list)):
            self.npy_files.append(self.npy_path + self.npy_file_list[i] + '.npy')
        self.len_files = len(self.npy_files)


    def prediction_publish(self, idx):
        clock = Clock()

        record = np.load(os.path.join(self.npy_path,self.npy_files[idx]))


        lidar = record[:,:,:5]

        # to perform prediction
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
        )

        norm_lidar = record

        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [norm_lidar],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]

        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        cloud = np.stack((x,y,z,i, label))

        header = Header()
        header.stamp = rospy.Time().now()
        header.frame_id = "velodyne_link"

        # point cloud segments
        msg_segment = self.create_cloud_xyzil32(header, cloud.T)

        # publish
        self._pub.publish(msg_segment)
        rospy.loginfo("Point cloud processed. Took %.6f ms.", clock.takeRealTime())


    # create pc2_msg with 5 fields
    def create_cloud_xyzil32(self, header, points):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
				  PointField('y', 4, PointField.FLOAT32, 1),
				  PointField('z', 8, PointField.FLOAT32, 1),
				  PointField('intensity', 12, PointField.FLOAT32, 1),
				  PointField('label', 16, PointField.FLOAT32, 1)]
        return pc2.create_cloud(header, fields, points)



if __name__ == '__main__':
    rospy.init_node('point_cloud_seg')

    npy_path = rospy.get_param('npy_path')
    npy_path_pred = rospy.get_param('npy_path_pred')
    npy_file_list = rospy.get_param('npy_file_list')

    pub_topic = rospy.get_param('pub_topic')
    checkpoint = rospy.get_param('checkpoint')
    gpu = rospy.get_param('gpu')

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
    'checkpoint', checkpoint,
    """Path to the model paramter file.""")
    tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

    npy_tensorflow_to_ros = NPY_TENSORFLOW_TO_ROS(pub_topic=pub_topic,
													   FLAGS=FLAGS,
													   npy_path=npy_path,
													   npy_file_list=npy_file_list)
