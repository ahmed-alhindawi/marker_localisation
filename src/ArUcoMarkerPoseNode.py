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
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from marker_localisation.msg import MarkerTagDetection, MarkerTagDetectionArray


class ArUcoBoardNode(object):
    def __init__(self):
        self._tf_broadcaster = TransformBroadcaster()
        self._tf_listener = TransformListener()
        self._cv_bridge = CvBridge()
        self._publish_tf = rospy.get_param("~publish_tf", default=True)
        _required_dict = rospy.get_param("~dictionary", default="original").lower()
        _allowed_dicts = ["ORIGINAL", "4x4", "5x5", "6x6"]

        if _required_dict == "original":
            self._dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        elif _required_dict == "4x4":
            self._dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        elif _required_dict == "5x5":
            self._dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        elif _required_dict == "6x6":
            self._dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        else:
            raise ValueError("Required ARUCO dictionary not available. Available ones are: {}".format(_allowed_dicts))

        self._marker_length = rospy.get_param("~marker_length", default=0.10)

        rospy.loginfo("Waiting for camera info messages")
        self.camera = image_geometry.PinholeCameraModel()
        cam_info_msg = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
        self.camera.fromCameraInfo(cam_info_msg)
        rospy.loginfo("...Received")

        rospy.Subscriber("/image", Image, self.publish_marker_transform)
        self._marker_pub = rospy.Publisher("/tags", MarkerTagDetectionArray, queue_size=10)

    def publish_marker_transform(self, cam_img):
        frame = self._cv_bridge.imgmsg_to_cv2(cam_img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # identify markers and
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, self._dict)

        if ids is not None and len(ids) > 0:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self._marker_length, self.camera.K, self.camera.D)

            tags = []
            for i, marker_id in enumerate(ids):
                trans = tvec[i][0]
                rot = rvec[i][0]
                angle = np.linalg.norm(rot)
                axis = rot.flatten() / angle

                quaternion = transformations.quaternion_about_axis(angle, axis)

                tag_corners = corners[i].reshape(4, 2)
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
                tag.id = marker_id
                tag.corners2d = [c1, c2, c3, c4]
                tag.tag_size = self._marker_length
                tag.pose = PoseStamped()
                tag.pose.pose.position = Point(*trans)
                tag.pose.pose.orientation = Quaternion(*quaternion)
                tag.pose.header = cam_img.header

                tags.append(tag)

                if self._publish_tf:
                    self._tf_broadcaster.sendTransform(translation=trans,
                                                       rotation=quaternion,
                                                       time=rospy.Time(cam_img.header.stamp.secs, cam_img.header.stamp.nsecs),
                                                       child="ar_marker_{}".format(marker_id[0]),
                                                       parent=cam_img.header.frame_id)

            mtda = MarkerTagDetectionArray()
            mtda.detections = tags
            self._marker_pub.publish(mtda)


# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('aruco_marker_pose_publisher')

    # Create the camera publisher
    tap = ArUcoBoardNode()
    rospy.spin()
