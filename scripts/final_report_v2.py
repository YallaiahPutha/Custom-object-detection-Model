import os
import cv2
import pandas as pd
import argparse
from collections import namedtuple
import numpy as np
import warnings
from PIL import Image
from pathlib import Path
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

########################################################################## DEFINE SETTINGS ###########################################################
min_score_thresh = 0.50
min_iou_thresh = 0.50
output_directory='/home/ai-team/Object_detection_models/data/washroom/reports/'
# input files
prediction_file_name = "/home/ai-team/Object_detection_models/data/washroom/results/eval_results.csv"
ground_truth_file_name = "/home/ai-team/Object_detection_models/data/washroom/test_gt.csv"
objs_list = "/home/ai-team/Object_detection_models/data/washroom/classes.txt"
# output files
output_prediction_file_name = output_directory+'filtered_predictions.csv'
predictions_summary_csv_path = output_directory+'predictions_summary.csv'
output_result_summary_file_name = output_directory+'metrics_summary.csv'
final_obj_metrics_file_name = output_directory+'obj_metrics_summary.csv'
cf_file_name = output_directory+'cf.csv'
iou_path = output_directory+'iou/'
########################################################################## DEFINE SETTINGS ###########################################################
predictions_summary_columns = ['image', 'object_id', 'predicted_box', 'predicted_class', 'confidence', 'gt_box', 'gt_class', 'iou']
# Path(args.out_dir).mkdir(parents=True, exist_ok=True)
Path(output_directory).mkdir(parents=True, exist_ok=True)
Path(iou_path).mkdir(parents=True, exist_ok=True)

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    # expects bbox as (xmin, ymin, xmax, ymax)
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def group_by(df, group):
    data = namedtuple('data', ['index_column', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

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

# pred_df = pd.read_csv(args.predictions)
pred_df = pd.read_csv(prediction_file_name)
result_summary_df = pred_df[pred_df['confidence'] >= min_score_thresh] # Filter out the predictions having confidence less than the threshold
result_summary_df.to_csv(output_prediction_file_name)

# inputs = pd.read_csv(args.ground_truth)
inputs = pd.read_csv(ground_truth_file_name)

with open(objs_list) as f:
    objs = f.read().splitlines()
print(objs)
print(len(objs))
print(type(objs))
inputs_filtered = inputs[inputs['class'].isin(objs)]

true_pos={}
false_pos={}
false_neg={}
cf={}

for item in objs:
    true_pos[item]=0
    false_pos[item]=0
    false_neg[item]=0
    cf[item]=dict.fromkeys(objs,0)
    
predictions_df = result_summary_df.copy()
img_annots = group_by(inputs_filtered, 'filename')

predictions_summary = []
for index, row in predictions_df.iterrows():
    ground_truth = inputs[inputs['filename']==row['image']]
    pred_box = row['box'][1:-1].split(',')
    pred_box = [float(x) for x in pred_box]
    iou = []
    boxes = []
    predicted_classes = []
    for truth_index, truth_row in ground_truth.iterrows():
        # need boxes in the format [xmin, ymin, xmax, ymax]
        truth_box = [truth_row['xmin'], truth_row['ymin'], truth_row['xmax'], truth_row['ymax']]
        iou.append(bb_intersection_over_union(pred_box, truth_box))
        boxes.append(tuple(truth_box))
        predicted_classes.append(truth_row['class'])
        # predicted_classes.append(''.join(truth_row['class'].split(' ')))
    
    match_box = ''
    match_iou = ''
    match_class = ''
    if len(iou) >= 1:
        max_iou_idx = np.argmax(np.asarray(iou))
        if(iou[max_iou_idx]) >= min_iou_thresh:
            match_box = tuple(boxes[max_iou_idx])
            match_iou = iou[max_iou_idx]
            match_class = predicted_classes[max_iou_idx]
    else:
        match_box = 'No ground truth box found'
    predictions_summary.append((row['image'], row['object_id'], row['box'], row['predict_class'], row['confidence'], match_box, match_class, match_iou))


predictions_summary_df = pd.DataFrame(predictions_summary, columns=predictions_summary_columns)
predictions_summary_df.to_csv(predictions_summary_csv_path)


# Calculating TP, FP and FN
metric_summary = []
metric_summary_columns = ['image', 'True Positives', 'False Positives', 'False Negatives', 'ground_truth_count', 'predicted_count','ground_truth_labels', 'predicted_labels', "true_positive_labels", "less_iou_fn", "false_negative_no_true_match", "false_positive"]
count = 0

for img in img_annots: 

    image_path = img[0]
    gt_annotation_df = img[1]
    gt_total_obj = len(gt_annotation_df.index) #num of annotations per each image in ground truth
    predictions = result_summary_df[result_summary_df['image']==image_path]
    predict_total_obj = len(predictions.index) #num of annotations per each image in predictions
    tp = 0
    fn = 0
    tp_iou = 0
    predictions['matched'] = 'False'
    predictions.reset_index(drop=True, inplace=True)
    #print(f'Processing image: {image_path}')
    #print('Predictions')
    #print(predictions['matched'])
    #print(f'GT Objects: {gt_total_obj} Total predictions: {len(predictions.index)}')
    true_positive_labels = []
    less_iou_fn = []
    false_negative_no_true_match = []
    false_positive = []
    gt_labelss=[]
    pd_labels=list(predictions['predict_class'])
    
    for gt_index, gt in gt_annotation_df.iterrows(): #processing ground truth df for one image
    
        temp_iou_list = []
        gt_label = gt['class']
        gt_labelss.append(gt_label)
        gt_box = [gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']]
        gt_flag = "False"
        
        for annot_index, annot_row in predictions.iterrows(): #processing pred df for one image
            predict_label = annot_row['predict_class']
            temp_iou_list.append(-1) #verify
            
            #insights start
            predict_box = annot_row['box'][1:-1].split(',')
            predict_box = [float(x) for x in predict_box]
            iou = bb_intersection_over_union(predict_box, gt_box)
            
            if iou > min_iou_thresh:
                cf[gt_label][predict_label] += 1
            
            
            #ends
            
            if (predict_label == gt_label) and (annot_row['matched'] == 'False'):
                predict_box = annot_row['box'][1:-1].split(',')
                predict_box = [float(x) for x in predict_box]
                iou = bb_intersection_over_union(predict_box, gt_box)
                temp_iou_list[-1] = iou
                
        #print(f'IOU List: {temp_iou_list}')
        
        if len(temp_iou_list) >= 1:
            max_iou_idx = np.argmax(np.asarray(temp_iou_list))
            
            if temp_iou_list[max_iou_idx] >= min_iou_thresh:
                tp += 1
                #predictions.at[max_iou_idx,'matched'] = 'True'
                predictions.loc[max_iou_idx, "matched"] = 'True'
                gt_flag = "True"
                #print(f'{gt_box} matched with the prediction at {max_iou_idx}')
                #print(f'Match Series: {predictions["matched"]}')
                true_positive_labels.append(gt_label)
                true_pos[gt_label] += 1
                
        if gt_flag == "False":

            for annot_index, annot_row in predictions.iterrows(): #processing pred df for one image
                predict_label = annot_row['predict_class']
                temp_iou_list.append(-1) #verify
                
                if (predict_label == gt_label) and (annot_row['matched'] == 'False'):
                    predict_box = annot_row['box'][1:-1].split(',')
                    predict_box = [float(x) for x in predict_box]
                    iou = bb_intersection_over_union(predict_box, gt_box)
                    temp_iou_list[-1] = iou
                    
            #print(f'IOU List: {temp_iou_list}')
            
            if len(temp_iou_list) >= 1:
                max_iou_idx = np.argmax(np.asarray(temp_iou_list))
                
                if temp_iou_list[max_iou_idx] >= 0.5:
                    tp += 1
                    #predictions.at[max_iou_idx,'matched'] = 'True'
                    predictions.loc[max_iou_idx, "matched"] = 'True'
                    
                    ##################
                    im_w, im_h, image_np = load_image_into_numpy_array(gt['filename'])
                    write_image = image_np.copy()
                    max_box = predictions.loc[max_iou_idx, "box"][1:-1].split(',')
                    max_box = [float(x) for x in max_box]
                    write_image = cv2.rectangle(write_image, (int(max_box[0]),int(max_box[1])), (int(max_box[2]),int(max_box[3])), (255,0,0), 1)
                    cv2.putText(write_image, gt_label, (int(max_box[0]), int(max_box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
                    
                    write_image = cv2.rectangle(write_image, (int(gt_box[0]),int(gt_box[1])), (int(gt_box[2]),int(gt_box[3])), (0,255,0), 1)
                    cv2.putText(write_image, gt_label, (int(gt_box[0]), int(gt_box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                    
                    i_path = gt['filename'].split('/')[-1]
                    out_path = f'{iou_path}/{i_path}'
                    (Image.fromarray(write_image)).save(out_path)
                    ##################
                    
                    #print(f'{gt_box} matched with the prediction at {max_iou_idx}')
                    #print(f'Match Series: {predictions["matched"]}')
                    true_positive_labels.append(gt_label)
                    true_pos[gt_label] += 1
                    
                else:
                    #modify
                    fn += 1
                    false_neg[gt_label] += 1
                    
                    if temp_iou_list[max_iou_idx] != -1:
                        less_iou_fn.append([gt_label,temp_iou_list[max_iou_idx]])  
                    else:
                        false_negative_no_true_match.append(gt_label)
                        
                    #print('No match!')
            else:
                #modify
                fn += 1
                false_neg[gt_label] += 1
                false_negative_no_true_match.append(gt_label)
                
    #modify        
    fp = predict_total_obj - tp
    false_positive=list(predictions.loc[predictions['matched']=="False"]["predict_class"])
    for item in list(predictions.loc[predictions['matched']=="False"]["predict_class"]):
        #print(item)
        false_pos[item] +=1
    metric_summary.append((image_path, tp, fp, fn, len(gt_labelss), len(pd_labels), gt_labelss, pd_labels, true_positive_labels, less_iou_fn, false_negative_no_true_match, false_positive))
    print('count:',count)
    count = count + 1

final_obj_metrics=pd.DataFrame([true_pos,false_pos,false_neg],index=["true_pos","false_pos","false_neg"]).T
final_obj_metrics["precision"]=final_obj_metrics["true_pos"]/(final_obj_metrics["true_pos"]+final_obj_metrics["false_pos"])
final_obj_metrics["recall"]=final_obj_metrics["true_pos"]/(final_obj_metrics["true_pos"]+final_obj_metrics["false_neg"])

final_obj_metrics.to_csv(final_obj_metrics_file_name)

metric_summary_df = pd.DataFrame(metric_summary, columns=metric_summary_columns)
metric_summary_df.to_csv(output_result_summary_file_name)

cf_summary=pd.DataFrame(cf)
cf_summary.to_csv(cf_file_name)
