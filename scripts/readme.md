## STEP 1 Convert the rosbags to rgb and depth frames
### The rosbags library only works with python 3.8.10 or less so create an environment first.
### The rgb and depth images are stores in OUPUT-DIRECTORY/rgb and OUTPUT_DIRECTORY/depth respectively.
### Bundle track uses the first pose from annotated_poses directory for tracking, rest of the poses can be any random matrix, 
### this code stores identity matrix at all the OUTPUT_DIRECTORY/annotated_poses/*.txt. Change the first pose to the actual 
### location of object to get absolute tracking.

In rosbag_reader environment(python 3.8.10)
```
conda create -n rosbag_reader python=3.8.10
conda activate rosbag_reader
pip install rosbags
python3 rosbag2_reader.py $PATH-TO-ROSBAG $PATH-TO-OUTPUT-DIRECTORY
```

## STEP 2 Generate initial binary mask for the object of interest in the first rgb image.
### First install segment anything from here https://github.com/facebookresearch/segment-anything, preferably make a new
### conda environment and do the installations inside so that it does not mess with your cuda versions.

```
conda create -n BundleTrack
conda activate BundleTrack
```
### Follow installation steps for SAM

```
python generate_initial_mask.py
```
### Cut and paste the initial_mask to the OUTPUT-DIRECTORY from OUTPUT-DIRECTORY/rgb

## STEP 3 Generate the greyscale masks for all the images using transductive.pytorch video segmentation network. 
###  Follow the steps in BundleTrack repo, example implementation after getting in Bundle track docker

```
python transductive-vos.pytorch/run_video.py --img_dir /home/sarthak/dataset/YCBInEOAT/BT_object_test/rgb --init_mask_file /home/sarthak/dataset/YCBInEOAT/BT_object_test/frame_0000000_init_mask.png --mask_save_dir /home/sarthak/dataset/YCBInEOAT/BT_object_test/masks
```

## STEP 4 Finally, run the Bundle Track code to get output data stored in /tmp/BundleTrack (can be changed)
### Example implementation
```
python scripts/run_ycbineoat.py --data_dir /home/sarthak/dataset/YCBInEOAT/BT_object_test --port 5555
```

## Visualization: To visualize the bounding boxes
### Copy paste the poses from the docker to the host machine, change the relevant paths in plot_bbox.py and run the code.