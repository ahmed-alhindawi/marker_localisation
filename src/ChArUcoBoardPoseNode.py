#!/usr/bin/env python

from __future__ import print_function

import rospy
from tf import TransformListener, TransformBroadcaster, LookupException, ExtrapolationException
from tf2_py import TransformException
from tf import transformations
import rospkg
import cv2.aruco as aruco
import cv2
from cv_bridge import CvBridgeError, CvBridge
import image_geometry
import os
import yaml
import numpy as np

from sensor_msgs.msg import Image, CameraInfo


class ChArUcoBoardNode(object):
    def __init__(self):
        self._tf_broadcaster = TransformBroadcaster()
        self._tf_listener = TransformListener()
        self._cv_bridge = CvBridge()

        rospy.loginfo("Waiting for camera info messages")
        self.camera = image_geometry.PinholeCameraModel()
        cam_info_msg = rospy.wait_for_message(
            rospy.get_param("~camera_info_topic", "camera_info"), CameraInfo, timeout=None)
        self.camera.fromCameraInfo(cam_info_msg)
        rospy.loginfo("...Received")

        self._world_quat = np.array([0.0, 1.0, 0.0, 0.0])

        board_config_path = os.path.join(rospkg.RosPack().get_path("visual_attention_setup"), "model", "charuco_board.yaml")
        with open(board_config_path, 'r') as stream:
            try:
                board_config = yaml.load(stream)
                self._name = board_config["name"]
                self._dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
                self._board = aruco.CharucoBoard_create(board_config['columns'], board_config['rows'],
                                                        board_config['square_size'], board_config['marker_size'],
                                                        self._dict)

                rospy.Subscriber("image", Image, self.publish_marker_transform)
            except yaml.YAMLError as exc:
                print(exc)

    def publish_marker_transform(self, cam_img):
        frame = self._cv_bridge.imgmsg_to_cv2(cam_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # identify markers and
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, self._dict)

        if ids is not None:
            ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self._board)

            # if there are enough corners to get a reasonable result
            if ret > 7:
                aruco.drawDetectedCornersCharuco(frame, ch_corners, ch_ids, (0, 0, 255))
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, self._board, self.camera.K,
                                                                self.camera.D)

            # if a pose could be estimated
            if retval:
                angle = np.linalg.norm(rvec)
                axis = rvec.flatten() / angle

                quaternion = transformations.quaternion_about_axis(angle, axis)

                self._tf_broadcaster.sendTransform(translation=tvec, rotation=quaternion, time=cam_img.header.stamp,
                                                   child=self._name, parent=cam_img.header.frame_id)


# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('multiple_average_transform_publisher')

    # Create the camera publisher
    tap = ChArUcoBoardNode()
    rospy.spin()
