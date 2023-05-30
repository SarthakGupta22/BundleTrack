# Authors: Sarthak Gupta, Bowen Wen
# Contact: gupta.sart@northeastern.edu
# Created in 2023

# Copyright (c) Northeastern University(River LAB), 2023 All rights reserved.

# This is a slight modification of the original code by Bowen Wen
# Inputs: RGB images, camera intrinsics(K), object point cloud(Optional) and detected poses from BundleTrack
# Output: A video containing bounding boxes around the detected object pose

import cv2
import numpy as np
import copy
import glob
import os

def draw_bbox(K, img, ob_in_cam, bbox, color_id, linewidth=2):
# These points are used to decide the boundaries of the bounding box
  def search_fit(points):
   '''
   @points: (N,3)
   '''
   min_x = min(points[:, 0])
   max_x = max(points[:, 0])
   min_y = min(points[:, 1])
   max_y = max(points[:, 1])
   min_z = min(points[:, 2])
   max_z = max(points[:, 2])
   return [min_x, max_x, min_y, max_y, min_z, max_z]
 
 def build_frame(min_x, max_x, min_y, max_y, min_z, max_z, frame_width, frame_height):
   bbox = []
   for i in np.arange(min_x, max_x, 1.0):
           bbox.append([i, min_y, min_z])
   for i in np.arange(min_x, max_x, 1.0):
           bbox.append([i, min_y, max_z])
   for i in np.arange(min_x, max_x, 1.0):
           bbox.append([i, max_y, min_z])
   for i in np.arange(min_x, max_x, 1.0):
           bbox.append([i, max_y, max_z])
 
   for i in np.arange(min_y, max_y, 1.0):
           bbox.append([min_x, i, min_z])
   for i in np.arange(min_y, max_y, 1.0):
           bbox.append([min_x, i, max_z])
   for i in np.arange(min_y, max_y, 1.0):
           bbox.append([max_x, i, min_z])
   for i in np.arange(min_y, max_y, 1.0):
           bbox.append([max_x, i, max_z])
 
   for i in np.arange(min_z, max_z, 1.0):
           bbox.append([min_x, min_y, i])
   for i in np.arange(min_z, max_z, 1.0):
           bbox.append([min_x, max_y, i])
   for i in np.arange(min_z, max_z, 1.0):
           bbox.append([max_x, min_y, i])
   for i in np.arange(min_z, max_z, 1.0):
           bbox.append([max_x, max_y, i])
   bbox = np.array(bbox)
   return bbox
 cam_cx = K[0,2]
 cam_cy = K[1,2]
 cam_fx = K[0,0]
 cam_fy = K[1,1]
 
 target_r = ob_in_cam[:3,:3]
 target_t = ob_in_cam[:3,3]*1000
 
 target = copy.deepcopy(bbox)
 limit = search_fit(target)

 bbox = build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5], frame_width, frame_height)
 bbox = np.dot(bbox, target_r.T) + target_t


 color = np.array([[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250], [189, 183, 107], [165, 42, 42], [0, 234, 0]])
 vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

 for tg in bbox:
   y = int(tg[0] * cam_fx / tg[2] + cam_cx)
   x = int(tg[1] * cam_fy / tg[2] + cam_cy)
 
   if x - linewidth < 0 or x + linewidth > frame_height-1 or y - linewidth < 0 or y + linewidth > frame_width-1:
     continue
 
   for xxx in range(x-linewidth+1, x+linewidth):
     for yyy in range(y-linewidth+1, y+linewidth):
       vis[xxx][yyy] = color[color_id]
 
 vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
 return vis


K = np.loadtxt('/home/sarthak/dataset/YCBInEOAT/BT_object_test/cam_K.txt')

# These points should ideally be a point cloud of the object, right now two random points are used to set the bounding box size of 20x20x20 pixels
points = np.array([[-10, -10, -10], [10, 10, 10]])

# These values should match the original size of the RGB image
frame_width = 640 
frame_height = 360 #480

video_out = cv2.VideoWriter('/home/sarthak/outputs/ycbineoat/BT_object_test/outputs.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
for file in sorted(os.listdir("/home/sarthak/dataset/YCBInEOAT/BT_object_test/rgb")):
        img_path = os.path.join("/home/sarthak/dataset/YCBInEOAT/BT_object_test/rgb", file)
        obj_text_file_name = file[0:-4] + ".txt"
        obj_pose_path = os.path.join("/home/sarthak/outputs/ycbineoat/BT_object_test/poses", obj_text_file_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (frame_width, frame_height), interpolation = cv2.INTER_AREA)
        try:
                ob_in_cam = np.loadtxt(obj_pose_path)
        except:
                print("Pose not found for the object in image", file[0:-4], ".png")
                continue

        out = draw_bbox(K, img.copy(), ob_in_cam, points, 1)
        # Write the frame into the file 'output.avi'
        video_out.write(out)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video_out.release()
 
# Closes all the frames
cv2.destroyAllWindows()