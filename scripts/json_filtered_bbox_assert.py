#python filter.py --input_json "/root/lab/prime_team_projects/od_projects/project2/annotations/project2_main_modified.json" --output_json "/root/lab/prime_team_projects/od_projects/project2/annotations/prop-250-modified-main.json" --categories "/root/lab/prime_team_projects/od_projects/project2/files/250list.txt"


import json
from pathlib import Path

anno_dct={}

class CocoFilter():
    #Filters the COCO dataset
    
    def _process_info(self):
        self.info = self.coco['info']
        
    def _process_licenses(self):
        self.licenses = self.coco['licenses']
    
        
    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()
        self.category_set = set()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']
            
            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
                self.category_set.add(category['name'])
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')
            
            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id} # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        self.image_height_upper = dict()
        self.image_width_upper = dict()
        self.image_height_lower = dict()
        self.image_width_lower = dict()
        self.image_area_upper_thresh = dict()
        self.image_area_lower_thresh = dict()
        for image in self.coco['images']:
            image_id = image['id']
            image_height = image['height']
            image_width = image['width']
            if image_id not in self.images:
                self.images[image_id] = image
                self.image_height_upper[image_id] = 0.925 * image_height
                self.image_width_upper[image_id] = 0.925 * image_width
                
                self.image_height_lower[image_id] = 0.02 * self.image_height_upper[image_id]
                self.image_width_lower[image_id] = 0.02 * self.image_width_upper[image_id]
                
                self.image_area_upper_thresh[image_id] = self.image_height_upper[image_id] * self.image_width_upper[image_id]
                self.image_area_lower_thresh[image_id] = self.image_height_lower[image_id] * self.image_width_lower[image_id]
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')
                
    def _process_segmentations(self):
        self.segmentations = dict()
        #self.segmentation_dim = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            #ann_id = segmentation['id']
            bbox = segmentation['bbox']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
                #self.segmentation_dim[image_id] = []
            self.segmentations[image_id].append(segmentation)
            #self.segmentation_dim[image_id].append(bbox[2] * bbox[3])

    def _filter_categories(self):
        """ Find category ids matching args
            Create mapping from original category id to new category id
            Create new collection of categories
        """
        missing_categories = set(self.filter_categories) - self.category_set
        if len(missing_categories) > 0:
            print(f'Did not find categories: {missing_categories}')
            should_continue = input('Continue? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()

        self.new_category_map = dict()
        new_id = 0
        for key, item in self.categories.items():
            if item['name'] in self.filter_categories:
                self.new_category_map[key] = new_id
                new_id += 1

        self.new_categories = []
        for original_cat_id, new_id in self.new_category_map.items():
            new_category = dict(self.categories[original_cat_id])
            new_category['id'] = new_id
            self.new_categories.append(new_category)

    def _filter_annotations(self):
        """ Create new collection of annotations matching category ids
            Keep track of image ids matching annotations
        """
        self.new_segmentations = []
        self.new_image_ids = set()
        ann_count = 0
        for image_id, segmentation_list in self.segmentations.items():
            for segmentation in segmentation_list:
                original_seg_cat = segmentation['category_id']
                if segmentation['bbox'] is not None:
                    segmentation_dim = segmentation['bbox']
                    x = segmentation_dim[0]
                    y = segmentation_dim[1]
                    width = segmentation_dim[2]
                    height = segmentation_dim[3]
                    box_area = segmentation_dim[2] * segmentation_dim[3]
                    if original_seg_cat in self.new_category_map.keys() and box_area <= self.image_area_upper_thresh[image_id] and box_area >= self.image_area_lower_thresh[image_id]:
                        if x <= 0:
                            x=1
                            width = width - abs(x) -1
                        if y <= 0:
                            y=1
                            height = height - abs(y) -1
                        if x+width >= self.images[image_id]['width']:
                            width = self.images[image_id]['width'] - x - 1
                        if y+height >= self.images[image_id]['height']:
                            height = self.images[image_id]['height'] - y - 1
                        segmentation['bbox'] = [x, y, width, height]
                        new_segmentation = dict(segmentation)
                        cid=self.new_category_map[original_seg_cat]
                        new_segmentation['category_id'] = self.new_category_map[original_seg_cat]
                        if cid in anno_dct and anno_dct[cid]<2000:
                            self.new_segmentations.append(new_segmentation)
                            self.new_image_ids.add(image_id)
                            anno_dct[cid]+=1
                            ann_count += 1
                        elif cid not in anno_dct:
                            self.new_segmentations.append(new_segmentation)
                            self.new_image_ids.add(image_id)
                            anno_dct[cid]=1
                            ann_count += 1
                        #print(self.new_segmentations)
        print('Total annotations are - ', ann_count)

    def _filter_images(self):
        """ Create new collection of images
        """
        self.new_images = []
        img_count = 0
        for image_id in self.new_image_ids:
            self.new_images.append(self.images[image_id])
            img_count += 1
        print('Total images are - ', img_count)

    def main(self, args):
        # Open json
        self.input_json_path = Path(args.input_json)
        self.output_json_path = Path(args.output_json)
        
        with open(args.categories) as f:
            object_names = f.read().splitlines()
            
        print(object_names)    
        self.filter_categories = object_names

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()

        # Verify output path does not already exist
        if self.output_json_path.exists():
            should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()
        
        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        
        # Process the json
        print('Processing input json...')
        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

        # Filter to specific categories
        print('Filtering...')
        self._filter_categories()
        self._filter_annotations()
        self._filter_images()

        # Build new JSON
        new_master_json = {
            'info': self.info,
            'licenses': self.licenses,
            'images': self.new_images,
            'annotations': self.new_segmentations,
            'categories': self.new_categories
        }
        print(self.new_segmentations)

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Filtered json saved.')

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Instances JSON file to only include specified categories. "
    "This includes images, and annotations. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json",
        help="path to save the output json")
    parser.add_argument("-c", "--categories", dest="categories",
        help="List of category names separated by spaces, e.g. -c person dog bicycle")

    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)


