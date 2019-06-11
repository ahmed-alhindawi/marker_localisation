#!/usr/bin/env python

"""
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
"""

from __future__ import print_function

import rospy
from tf import TransformListener, TransformBroadcaster
from tf import transformations
import cv2.aruco as aruco
import cv2
from cv_bridge import CvBridge
import image_geometry
import numpy as np

from sensor_msgs.msg import Image, CameraInfo


class ArUcoBoardNode(object):
    def __init__(self):
        self._tf_broadcaster = TransformBroadcaster()
        self._tf_listener = TransformListener()
        self._cv_bridge = CvBridge()
        self._world_quat = np.array([0.0, 1.0, 0.0, 0.0])
        self._dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self._marker_length = rospy.get_param("~marker_length", 0.10)

        rospy.Subscriber("image", Image, self.publish_marker_transform)

        rospy.loginfo("Waiting for camera info messages")
        self.camera = image_geometry.PinholeCameraModel()
        cam_info_msg = rospy.wait_for_message(
            rospy.get_param("~camera_info_topic", "camera_info"), CameraInfo, timeout=None)
        self.camera.fromCameraInfo(cam_info_msg)
        rospy.loginfo("...Received")

    def publish_marker_transform(self, cam_img):
        frame = self._cv_bridge.imgmsg_to_cv2(cam_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # identify markers and
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, self._dict)

        if ids is not None and len(ids) > 0:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self._marker_length, self.camera.K, self.camera.D)

            for i, marker_id in enumerate(ids):
                trans = tvec[i][0]
                rot = rvec[i][0]
                angle = np.linalg.norm(rot)
                axis = rot.flatten() / angle

                quaternion = transformations.quaternion_about_axis(angle, axis)
                self._tf_broadcaster.sendTransform(translation=trans, rotation=quaternion, time=cam_img.header.stamp,
                                                   child="ar_marker_{}".format(marker_id[0]),
                                                   parent=cam_img.header.frame_id)


# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('multiple_average_transform_publisher')

    # Create the camera publisher
    tap = ArUcoBoardNode()
    rospy.spin()
