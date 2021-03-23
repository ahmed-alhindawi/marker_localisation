#!/usr/bin/env python

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
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from marker_localisation.msg import MarkerTagDetection


class ChArUcoBoardNode(object):
    def __init__(self):
        self._tf_broadcaster = TransformBroadcaster()
        self._tf_listener = TransformListener()
        self._cv_bridge = CvBridge()
        self._publish_tf = rospy.get_param("~publish_tf", default=True)
        self._last_rvev = None
        self._last_tvec = None

        rospy.loginfo("Waiting for camera info messages")
        self.camera = image_geometry.PinholeCameraModel()
        cam_info_msg = rospy.wait_for_message(
            rospy.get_param("~camera_info_topic", "/camera_info"), CameraInfo, timeout=None)
        self.camera.fromCameraInfo(cam_info_msg)
        rospy.loginfo("...Received")

        _required_dict = rospy.get_param("~dictionary", default="original").lower()

        if _required_dict == "original":
            self._dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        elif _required_dict == "4x4":
            self._dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        elif _required_dict == "5x5":
            self._dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        elif _required_dict == "6x6":
            self._dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        else:
            raise ValueError("Required CHARUCO dictionary not available. Available ones are: {}".format(
                ["ORIGINAL", "4x4", "5x5", "6x6"]))

        default_config = {
            'name': 'board1',
            'columns': 3,
            'rows': 3,
            'square_size': 0.05,
            'marker_size': 0.0395
        }

        self._board_config = rospy.get_param("~board_configuration", default=default_config)

        assert self._board_config is not None, "Board Configuration must be present"
        assert self._dict is not None, "Dictionary must be present"

        self._board = aruco.CharucoBoard_create(self._board_config['columns'],
                                                self._board_config['rows'],
                                                self._board_config['square_size'],
                                                self._board_config['marker_size'],
                                                self._dict)

        rospy.Subscriber(rospy.get_param("~camera_image_topic", default="/image"), Image, self.publish_marker_transform)
        self._marker_pub = rospy.Publisher("/tags", MarkerTagDetection, queue_size=10)

    def publish_marker_transform(self, cam_img):
        frame = self._cv_bridge.imgmsg_to_cv2(cam_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # identify markers and
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, self._dict)

        if ids is not None and len(ids) > 0:
            ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self._board, self.camera.K, self.camera.D)
            # if there are enough corners to get a reasonable result
            if ret > 3:
                use_guess = self._last_tvec is not None and self._last_rvev is not None
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, self._board, self.camera.K,
                                                                    self.camera.D, rvec=self._last_rvev, tvec=self._last_tvec, useExtrinsicGuess=use_guess)

                # if a pose could be estimated
                if retval:
                    self._last_tvec = tvec
                    self._last_rvev = rvec
                    angle = np.linalg.norm(rvec)
                    axis = rvec.flatten() / angle

                    quaternion = transformations.quaternion_about_axis(angle, axis)

                    tag_corners = ch_corners.reshape(ret, 2)
                    c1 = Point()
                    c1.x = tag_corners[0][0]
                    c1.y = tag_corners[0][1]

                    c2 = Point()
                    c2.x = tag_corners[1][0]
                    c2.y = tag_corners[1][1]

                    c3 = Point()
                    c3.x = tag_corners[2][0]
                    c3.y = tag_corners[2][1]

                    c4 = Point()
                    c4.x = tag_corners[3][0]
                    c4.y = tag_corners[3][1]

                    tag = MarkerTagDetection()
                    tag.id = self._board_config["name"]
                    tag.corners2d = [c1, c2, c3, c4]
                    tag.tag_size = self._board_config["marker_size"]
                    tag.pose = PoseStamped()
                    tag.pose.pose.position = Point(*tvec)
                    tag.pose.pose.orientation = Quaternion(*quaternion)
                    tag.pose.header = cam_img.header
                    self._marker_pub.publish(tag)

                    if self._publish_tf:
                        self._tf_broadcaster.sendTransform(translation=tvec,
                                                           rotation=quaternion,
                                                           time=cam_img.header.stamp,
                                                           child=self._board_config["name"],
                                                           parent=cam_img.header.frame_id)


# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('ChArUco board pose node')

    # Create the camera publisher
    ChArUcoBoardNode()
    rospy.spin()
