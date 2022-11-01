# python get_predictions.py --test_csv_path="/data1/root/lab/prime_team_projects/od_projects/project8/PW/files/pw_test.csv" --cfg_path="/data1/root/lab/prime_team_projects/od_projects/project8/PW/config/faster_rcnn_resnet152_v1_640x640_coco17_1.config" --ckpt_path="/data1/root/lab/prime_team_projects/od_projects/project8/PW/results/ckpt-101" --labels="/data1/root/lab/prime_team_projects/od_projects/project8/PW/files/pw.pbtxt" --out_dir=/data1/root/lab/prime_team_projects/od_projects/project8/PW/evaluation/ --conf_thr=0.1 


import time
import tensorflow as tf
import sys
import os
import argparse
import pandas as pd
from collections import namedtuple

'''print(sys.path)
sys.path.remove('/root/lab/TensorFlow/models/research')
sys.path.remove('/root/lab/TensorFlow/models/research/slim')
print(sys.path)'''

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import pathlib
import warnings
import six
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

parser = argparse.ArgumentParser(description='Processes the predictions and generates the evaluation metrics')
parser.add_argument('--test_csv_path', type=str, dest='test_csv_path', help='Path to csv file containing ground truth')
parser.add_argument('--cfg_path', type=str, dest='cfg_path', help='Path to the config file')
parser.add_argument('--ckpt_path', type=str, dest='ckpt_path', help='Path to the checkpoint')
parser.add_argument('--labels', type=str, dest='labels', help='Path to the labels.txt')
parser.add_argument('--out_dir', type=str, help='Output root directory', dest='out_dir')
parser.add_argument('--conf_thr', type=str, help='Output root directory', dest='conf_thr')

args = parser.parse_args()

min_score_thresh = float(args.conf_thr)

pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
result_csv_path = f'{args.out_dir}/eval_results.csv'
result_csv_path1 = f'{args.out_dir}/eval_results_indexed.csv'
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img = Image.open(path)
    im_width, im_height = img.size
    return (im_width, im_height, np.array(img))

def record_predictions(image, im_width, im_height, boxes, classes, scores, category_index, min_score_thresh=0.5):
    records = []
    for i in range(boxes.shape[0]):
        box = None
        score = None
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            box = (round(xmin*im_width), round(ymin*im_height), round(xmax*im_width), round(ymax*im_height)) #changed order from (ymin,xmin,ymax,xmax)
            score = scores[i]
            if classes[i] in six.viewkeys(category_index):
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'Unknown'
            records.append((image, i, box, class_name, score))
    return records   

def group_by(df, group):
    data = namedtuple('data', ['index_column', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(args.cfg_path)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(args.ckpt_path).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


category_index = label_map_util.create_category_index_from_labelmap(args.labels,
                                                                    use_display_name=True)

result_list = []
result_columns = ['image', 'object_id', 'box', 'predict_class', 'confidence' ]

inputs = pd.read_csv(args.test_csv_path)

img_annots = group_by(inputs, 'filename')
count = len(img_annots)
for i, img in enumerate(img_annots):

    try:
     
        image_path = img[0]
        orig_label = img[1]['class'].iloc[0]
        print('Running inference for {}... '.format(image_path))
        count -= 1
        print('Remaining Images Count {}... '.format(count))
        im_w, im_h, image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
            
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        #print(detections['detection_classes'])
    
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
    
        result_list.extend(record_predictions(image_path, im_w, im_h, detections['detection_boxes'], 
                detections['detection_classes']+label_id_offset, 
                detections['detection_scores'],category_index, min_score_thresh=min_score_thresh))
                
        result_list2 = record_predictions(image_path, im_w, im_h, detections['detection_boxes'], 
                detections['detection_classes']+label_id_offset, 
                detections['detection_scores'],category_index, min_score_thresh=min_score_thresh)
        
        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.5,
                agnostic_mode=False)
        
        out_obj_folder = f'{args.out_dir}/{orig_label}'
        pathlib.Path(out_obj_folder).mkdir(parents=True, exist_ok=True)
        fn = image_path.split('/')[-1]
        out_path = f'{out_obj_folder}/{fn}' 
        (Image.fromarray(image_np_with_detections)).save(out_path)
        print(result_list2)
        #saving
        result_summary_df2 = pd.DataFrame(result_list2, columns=result_columns)
        if i==0:
            result_summary_df2.to_csv(result_csv_path, index=False)
        else:
            result_summary_df2.to_csv(result_csv_path, mode = 'a', index=False, header=False)

    except:
        #result_csv_path = f'{args.out_dir}/eval_results.csv'
        #result_summary_df = pd.DataFrame(result_list, columns=result_columns)
        #result_summary_df.to_csv(result_csv_path)
        result_summary_df2 = pd.DataFrame(result_list2, columns=result_columns)
        if i==0:
            result_summary_df2.to_csv(result_csv_path, index=False)
        else:
            result_summary_df2.to_csv(result_csv_path, mode = 'a', index=False, header=False)
        with open('inference_error_status.txt', 'a') as f:
            f.write(str(image_path))
            f.write("\n")
  
eval_df=pd.read_csv(result_csv_path)
eval_df.to_csv(result_csv_path1)    
#result_csv_path = f'{args.out_dir}/eval_results.csv'
#result_summary_df = pd.DataFrame(result_list, columns=result_columns)
#result_summary_df.to_csv(result_csv_path)
