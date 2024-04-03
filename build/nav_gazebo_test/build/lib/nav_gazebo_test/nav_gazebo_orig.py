import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np


W, H = 1280, 800
fov_horizontal = 1.2217  # radians, approximately 60 degrees

# Calculate fx and fy using the horizontal FoV and assuming square pixels
fx = fy = W / (2 * np.tan(fov_horizontal / 2))

# The optical center is at the center of the image
cx, cy = W / 2, H / 2

# Your distortion coefficients are as provided
k1, k2, p1, p2, k3 = -0.25, 0.12, -0.00028, -0.00005, 0.0


class ArUcoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()

        # Camera matrix and distortion coefficients
        self.camera_matrix = np.array([[fx, 0, cx],
                                       [0, fy, cy],
                                       [0, 0, 1]])  # fx, fy, cx, cy need to be defined based on your camera
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # k1, k2, p1, p2, k3 should be your camera's distortion coefficients

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Undistort the image
        frame_undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        # ArUco marker detection on the undistorted image
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # If marker ID 0 is detected
        if ids is not None and 0 in ids:
            index_of_id_0 = np.where(ids == 0)[0]
            corners_of_id_0 = [corners[i] for i in index_of_id_0]
            aruco.drawDetectedMarkers(frame_undistorted, corners_of_id_0)

            for corner in corners_of_id_0:
                c = corner[0]
                center = c.mean(axis=0)
                position_text = f"Marker 0 Position: ({center[0]:.2f}, {center[1]:.2f})"
                cv2.putText(frame_undistorted, position_text, (int(center[0] + 20), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                vector = c[1] - c[0]
                angle = np.arctan2(vector[1], vector[0])
                heading_text = f"Marker 0 Heading: {np.degrees(angle):.2f} degrees"
                cv2.putText(frame_undistorted, heading_text, (int(center[0] + 20), int(center[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the undistorted frame
        cv2.imshow("ArUco Marker 0 Detection", frame_undistorted)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArUcoDetector()
    rclpy.spin(aruco_detector)
    aruco_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
