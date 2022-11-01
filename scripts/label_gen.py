"""l=["Backpack","Beetroot","Cauliflower","Chair","EggPlant","Idlytray","Knife","Lock","Shoes","Slippers","Bucket","Rain_Coat","Punching_Machine","Marker_Pen","Clothes_Hanger","Umbrella","Stapler","Lunch_Box","Induction_Stove","Inverter_Battery","CCTV","Capsicum","Ginger","Garlic","Onions","Scissors","Comb",
"Carrom_Board","Cup","Flowerpot"]"""

import json

f = open("/home/ai-team/Object_detection_models/data/washroom/train.json",)
data = json.load(f)
l=[]
for dictt in data['categories']:
    l.append(dictt['name'])

for i in range(len(l)):
    a=dict()
    a["id"]=i+1
    a["name"]=l[i]
    string=str(a).replace(", ","\n")
    string=string.replace("'id'","  id")
    string=string.replace("'name'","  name")
    string=string.replace("{","")
    string=string.replace("}","")
    with open('/home/ai-team/Object_detection_models/data/washroom/label_map.pbtxt', 'a') as f:
        f.write("item {")
        f.write("\n")
        f.write(string)
        f.write("\n")
        f.write("}")
        f.write("\n\n")