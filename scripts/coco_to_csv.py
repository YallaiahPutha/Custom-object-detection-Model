#python get_predictions.py --test_csv_path="/root/lab/prime_team_projects/prop-1000/files/prop1000_test_gt.csv" --cfg_path="/root/lab/prime_team_projects/prop-1000/files/pipeline_frcnn_project2_1000_i2.config" --ckpt_path="/root/lab/prime_team_projects/prop-1000/results/new/frcnn2/ckpt-95" --labels="/root/lab/prime_team_projects/prop-1000/files/1000label.pbtxt" --out_dir=/root/lab/prime_team_projects/prop-1000/pred/ --conf_thr=0.5                   
#
import json
import argparse
import funcy
import os
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Cleans annotation file by removing annotations for non existing images')
parser.add_argument('--annotations', type=str, dest='annotations', help='Path to COCO annotations file.')
parser.add_argument('--out', type=str, help='Output file', dest='out')
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')
parser.add_argument('--data_root', type=str, help='root directory for images', dest='data_root')

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def re_structure(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    image_idx = [*range(len(image_ids))]
    new_img_id = {id:idx for (id,idx) in zip(image_ids, image_idx)}
    for image in images:
        image['id'] = new_img_id[image['id']]
    
    i = 0
    for annot in annotations:
        annot['image_id'] = new_img_id[annot['image_id']]
        annot['id'] = i
        i += 1
    
    return annotations, images

def main(args):
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    annots = []
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)

        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']


        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)        
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        #img_index = {f['file_name']:int(f['id']) for f in images}
        class_dict = {int(c['id']):c['name'] for c in categories}
        img_dict ={int(f['id']):f for f in images}

        #bbox -> x,y,width, height
        for annotation in annotations:
            img_id = annotation['image_id']
            class_id = annotation['category_id']
            #filepath_split = img_dict[img_id]["file_name"].split('/')
            #filepath = f"{'/'.join(filepath_split[:-1])}/{filepath_split[-1]}"
            #filepath = f"{'/'.join(filepath_split[0])}/{filepath_split[-1]}" #modifications
            filepath = img_dict[img_id]["file_name"]
            if annotation['bbox'] != None:
                record = (f'{args.data_root}/{filepath}', img_dict[img_id]['width'], img_dict[img_id]['height']
                            ,class_dict[class_id]
                            ,annotation['bbox'][0] #xmin
                            ,annotation['bbox'][1]  #ymin
                            ,annotation['bbox'][0] + annotation['bbox'][2]  #xmax
                            ,annotation['bbox'][1] + annotation['bbox'][3]) #ymax
            annots.append(record)
        df = pd.DataFrame(annots, columns=column_name)

        df.to_csv(args.out)

        #cleaned_annot_path = '{}_clean.json'.format('.'.join((args.annotations).split('.')[:-1]))


if __name__ == "__main__":
    main(args)