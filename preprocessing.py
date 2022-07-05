"""Extracting images according to annotations from given videos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import os
import pathlib
from PIL import Image

import cv2
import skvideo.io
from smile import app
from smile import flags
from smile import logging

flags.DEFINE_string("data_path",
                    "/mnt/data/m2cai/m2cai_tool/train_dataset",
                    "Data path of surgical videos.")
flags.DEFINE_string("annot_path",
                    "/mnt/data/m2cai/m2cai_tool/train_dataset",
                    "Annotation path of surgical videos.")
flags.DEFINE_string("target_path",
                    "/mnt/data/m2cai/m2cai_tool/images/train",
                    "Path to save the images.")
flags.DEFINE_boolean("has_header", True,
                     "If the annotation file has a header line.")
flags.DEFINE_boolean("resize", True, "If to resize the image.")
flags.DEFINE_integer("resize_height", 224,
                     "Height to resize the image from video.")
flags.DEFINE_integer("resize_width", 224,
                     "Width to resize the image from video.")

FLAGS = flags.FLAGS

def get_cropped_idx(img):
    """Given an image as np array format, crop the non-black-border part.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY) # 32 is magic.
    idx = np.where(thresh == 255)
    x_start = np.min(idx[0])
    x_end = np.max(idx[0]) + 1
    y_start = np.min(idx[1])
    y_end = np.max(idx[1]) + 1
    return x_start, x_end, y_start, y_end

def extract_images_from_all_videos():
    """ Extract frame images with annotations from all the videos in a path.
    """
    # Get all videos.
    current_files = glob.glob(os.path.join(FLAGS.data_path, "*.mp4"))
    current_files.sort()
    assert len(current_files) > 0, "No video file has been found."
    logging.info("%d videos have been found." % len(current_files))
    if not os.path.isdir(FLAGS.target_path):
        pathlib.Path(FLAGS.target_path).mkdir(parents=True, exist_ok=True)
    logging.info("Extracting.")
    for each_file in current_files:
        logging.info("Extracting Training Images from video %s."
                     % each_file.strip().split("/")[-1])
        annot_path = glob.glob(
                        os.path.join(FLAGS.annot_path,
                                     each_file.split("/")[-1].split(".")[0] + \
                                        "*.txt"))
        assert len(annot_path) is 1, "Only one annotation file should be found!"
        annot_path = annot_path[0]
        logging.info(annot_path)
        assert os.path.isfile(annot_path), "Annotation file not exists."
        target_path = os.path.join(FLAGS.target_path,
                                   each_file.split('/')[-1].split('.')[0])
        if not os.path.isdir(target_path):
            pathlib.Path(target_path).mkdir(parents=True, exist_ok=True)
        extract_frames_from_video(each_file, annot_path, target_path,
                                  FLAGS.resize, FLAGS.resize_width,
                                  FLAGS.resize_height)

def extract_index_label_from_annotation(file_name):
    """Given an annotation file, extract all the groundtruth labels from it.
    """
    with open(file_name, "r") as f_reader:
        lines = f_reader.readlines()[1:]
    return [int(x.split()[0]) for x in lines], \
           [''.join(x.split()[1:]) for x in lines]

def extract_frames_from_video(video_path, annot_path, target_path, resize,
                              resize_width, resize_height, magic_idx=1000):
    """Extract all frames with annotations from a video.
    """
    if resize:
        assert resize_width > 0 and resize_height >0, \
            "Resized size should be positive"
    video_id = video_path.split("/")[-1].split(".")[0]
    indexes, labels = extract_index_label_from_annotation(annot_path)
    # if resize:
    output_dict = {"-sws_flags": "bilinear",
                   "-s": "%dx%d" % (480, 480)}
    # else:
    #     output_dict = None
    logging.info("Reading video.")
    vid = skvideo.io.vread(video_path, outputdict=output_dict)
    # vid = skvideo.io.vread(video_path)
    logging.info("Reading finished.")
    x_0, x_1, y_0, y_1 = get_cropped_idx(vid[magic_idx])
    logging.info("Saving frames.")
    for index, label in zip(indexes, labels):
        logging.info("Saving image %s",
                     os.path.join(target_path,
                                  "%s_%06d_%s.jpg" % (video_id, index, label)))
        current_img = vid[index][x_0:x_1, y_0:y_1]
        resized_img = cv2.resize(current_img, (resize_width, resize_height))
        Image.fromarray(resized_img).save(
            os.path.join(target_path,
                         "%s_%06d_%s.jpg" % (video_id, index, label)))

def main(_):
    """Main function to call."""
    extract_images_from_all_videos()


if __name__ == "__main__":
    app.run()
