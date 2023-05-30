# Authors: Sarthak Gupta
# Contact: gupta.sart@northeastern.edu
# Created in 2023

# Copyright (c) Northeastern University(River LAB), 2023 All rights reserved.

# Inputs: Ros2 bag file directory(.zstd file and metadata.yaml), output directory location
# Output: Writes depth and RGB frames along with annotated_poses(set to identity) at the depth image rate(currently 5Hz)

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from cv_bridge import CvBridge
import numpy as np
import argparse
import cv2
import os
import sys

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")

    args = parser.parse_args()

    bridge = CvBridge()
    count = 0
    # create reader instance and open for reading
    with Reader(args.bag_file) as reader:

        # iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/hololens/depth/image':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                min_timestamp = sys.maxsize
                best_rgb_msg = deserialize_cdr(rawdata, connection.msgtype)
                for connection2, timestamp2, rawdata2 in reader.messages():
                    if connection2.topic == '/hololens/rgb/image':
                        if abs(timestamp - timestamp2) < min_timestamp:
                            min_timestamp = abs(timestamp - timestamp2)
                            best_rgb_msg = deserialize_cdr(rawdata2, connection2.msgtype)
                
                depth_img = (bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")).copy()
                # Crop depth image to match the fov of rgb and depth, this is an approximation.
                depth_img = depth_img[20:200, :].copy()
                # depth_img = depth_img.astype(uint16) #already uint16
                cv2.imwrite(os.path.join(args.output_dir, "depth/frame_%07i.png" % count), depth_img)
                print("Wrote depth image %i" % count)  

                dim = (depth_img.shape[1], depth_img.shape[0])
                rgb_img = (bridge.imgmsg_to_cv2(best_rgb_msg, desired_encoding="passthrough")).copy()
                # resize rgb image to the depth image size(can also do opposite)
                rgb_img = (cv2.resize(rgb_img, dim, interpolation = cv2.INTER_AREA)).copy()
                cv2.imwrite(os.path.join(args.output_dir, "rgb/frame_%07i.png" % count), rgb_img)
                print("Wrote rgb image %i" % count)
                np.savetxt(os.path.join(args.output_dir, "annotated_poses/%07i.txt" % count), np.eye(4))
                print("Wrote annotated_pose %i" % count)
                count += 1

    return

if __name__ == '__main__':
    main()