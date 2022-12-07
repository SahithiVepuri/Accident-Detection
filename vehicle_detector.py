import numpy as np
import os
import urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2
import winsound

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = './mscoco_label_map.pbtxt'
# PATH_TO_LABELS = os.path.join('ObjectDetection\data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)

with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
      # screen = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # image_np = cv2.imdecode(screen,-1)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=3)
      
      for i, b in enumerate(boxes[0]):
        if classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 6 or classes[0][i] == 8:
          if scores[0][i] > 0.5:
            mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
            mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
            apx_distance = round((1-(boxes[0][i][3] - boxes[0][i][1]))**4, 1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800), int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 246, 14), 2)
            if apx_distance <= 0.5:
              if mid_x > 0.1 and mid_x < 0.6:
                cv2.putText(image_np, 'WARNING!', (int(mid_x*800)-50, int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                winsound.Beep(70, 50)


      # cv2.imshow('window',cv2.resize( image_np,(800, 600)))
      cv2.imshow('window', image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break