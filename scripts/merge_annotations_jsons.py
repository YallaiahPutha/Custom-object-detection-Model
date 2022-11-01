import json
import os
import pandas as pd
def annotations(annotate,annotate_count):
    for i in range(len(annotate)):
        aid[annotate[i]['id']]=annotate_count    
        annotate[i]['id']=annotate_count
        annotate_count=annotate_count+1
        annotate[i]['image_id']=iid[annotate[i]['image_id']]
        #bug in json file .. some objects are annotated as category id 0 .. and no object with category id 0 will be there in categories node of json
        if annotate[i]['category_id'] != 0:
            annotate[i]['category_id']=cid[annotate[i]['category_id']]      
        annotate[i]['segmentation'] = None
        #removed
        del annotate[i]['attributes']
        del annotate[i]['iscrowd']
    json_array['annotations'].extend(annotate)
    #print(json_array)
    return annotate_count

def images(img,image_count):
    for i in range(len(img)):
        iid[img[i]['id']]=image_count
        img[i]['id']=image_count
        image_count=image_count+1
        if img[i]['file_name'].endswith(("_jpg","_JPG","_jpeg","_JPEG")):
            img[i]['file_name']= dataset_folder + "images/"+ img[i]['file_name'].replace("_jpg",".jpg").replace("_JPG",".JPG").replace("_jpeg",".jpeg").replace("_JPEG",".JPEG")
        else:
            img[i]['file_name']= dataset_folder + "images/"+ img[i]['file_name']  
        #added
        img[i]['license'] = None 
        img[i]['video_id'] = None 
    json_array['images'].extend(img)
    return image_count
    
def categories(category,cat_count):
    for i in range(len(category)):        
        if category[i]['name'] not in list(id_mapping.keys()):                         
            cid[category[i]['id']] = cat_count
            id_mapping[category[i]['name']] = cat_count
            category[i]['id']=cat_count
            cat_count = cat_count+1     
        else:          
            cid[category[i]['id']]=id_mapping[category[i]['name']]     
            category[i]['id']=id_mapping[category[i]['name']]
        #added
        category[i]['supercategory'] = None
        category[i]['isthing'] = 0
        category[i]['color'] = []                         
    json_array['categories'].extend(category)
    return cat_count
    

id_mapping = {}


data = {}
data['info'] = {}
data['licenses']=[]
data['images']=[]
data['annotations']=[]
data['categories']=[]
data['info'] = {
        "description": "This is dataset.",
        "url": "https://superannotate.ai",
        "version": "1.0",
        "year": 2020,
        "contributor": "Superannotate AI",
        "date_created": "17/07/2020"
        }
data['licenses'].append({
            "url": "https://superannotate.ai",
            "id": 1,
            "name": "Superannotate AI"
        })
with open("/home/ai-team/Object_detection_models/data/washroom/new_veg_merge.json", 'w') as outfile:
    json.dump(data, outfile)

input_file=open("/home/ai-team/Object_detection_models/data/washroom/new_veg_merge.json") 
json_array = json.load(input_file)
image_count=len(json_array["images"])
annotate_count=len(json_array["annotations"])
cat_count=len(json_array["categories"])+1
input_file.close()

cid=dict()
main_directory = '/home/ai-team/Object_detection_models/'
dataset_folder = 'data/washroom/'
arr = os.listdir(main_directory + dataset_folder + "annotations/")
for i, item in enumerate(arr):
    try:
        indvidual_file= main_directory + dataset_folder + "annotations/"+item
        input_file=open(indvidual_file)
        js = json.load(input_file)
        #print(js)
        iid=dict()
        aid=dict()
        if(i ==0):
            print(i)
            cat_count=categories(js["categories"],cat_count)
        print('cat_count - ', cat_count)
        image_count=images(js["images"],image_count)
        print('image_count - ', image_count)
        print('annotate_count - ', annotate_count)
        annotate_count=annotations(js["annotations"],annotate_count)
        print('annotate_count after merge - ', annotate_count)
        input_file.close()
        
    except Exception as e:
        print(indvidual_file)
        print(e)
   
with open("/home/ai-team/Object_detection_models/data/washroom/new_veg_merge.json", 'w') as outfile:
    json.dump(json_array, outfile)
outfile.close()
