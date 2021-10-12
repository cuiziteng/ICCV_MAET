'''
Change Exdark dataset to xml format.
'''

import os
import os.path as osp
import random
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import cv2
import shutil
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='the place you store EXDark dataset')
opt = parser.parse_args()
print(opt)

data_dir = opt.data_dir
txt_path = data_dir + '/label'
xml_path = data_dir + '/xml'
img_path = data_dir + '/pic'
main_path = data_dir+ '/main'

class_names = ('Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
              'Cup', 'Dog', 'Motorbike', 'People', 'Table')

# Make dir to store xml type file.
def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)
mkdir(xml_path)
mkdir(main_path)

for class_name in class_names:
    mkdir(osp.join(xml_path, class_name))

# Delete the file for MAXOS system, which not use in this process.
def rmdir(path):
    if  osp.exists(path):
        shutil.rmtree(path)
rmdir(osp.join(txt_path ,'__MACOSX'))
rmdir(osp.join(img_path ,'__MACOSX'))



def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

def txt2xml(txt_path):
    """
    Change the txt file into xml format.
    """
    for file in findAllFile(txt_path):
        txt_whole_path = glob(osp.join(txt_path, '*', file))[0]
        img_whole_path = txt_whole_path.replace('label','pic').strip('.txt')
        xml_whole_path = txt_whole_path.replace('label','xml').replace('txt','xml')

        #create img id
        img_id = file.split('.')[0]
        print(img_id)

        img = cv2.imread(img_whole_path)
        #create root of XML file
        root = ET.Element('Annotation')
        ET.SubElement(root, 'filename').text = img_id
        sizes = ET.SubElement(root,'size')
        print(img.shape)
        height, width, channel = img.shape[0], img.shape[1], img.shape[2]
        ET.SubElement(sizes, 'width').text = str(width)
        ET.SubElement(sizes, 'height').text = str(height)
        ET.SubElement(sizes, 'depth').text = str(channel)

        f = open(txt_whole_path, "r")
        lines = f.readlines()
        #print('11111',lines)
        for line in lines[1:]:
            line = line.split(' ')
            cls = line[0]
            x1, y1, w, h = int(line[1]), int(line[2]), int(line[3]), int(line[4])
            #print(x1,y1,x2,y2)
            x2 = x1+w
            y2 = y1+h
            
            objects = ET.SubElement(root, 'object')
            #object class
            ET.SubElement(objects, 'name').text = cls    
            ET.SubElement(objects, 'pose').text = 'Unspecified'
            ET.SubElement(objects, 'truncated').text = '0'
            ET.SubElement(objects, 'difficult').text = '0'

            #bounding box location
            bndbox = ET.SubElement(objects,'bndbox')    
            ET.SubElement(bndbox, 'xmin').text = str(x1)
            ET.SubElement(bndbox, 'ymin').text = str(y1)
            ET.SubElement(bndbox, 'xmax').text = str(x2)
            ET.SubElement(bndbox, 'ymax').text = str(y2)

        tree = ET.ElementTree(root)
        tree.write(xml_whole_path, encoding='utf-8')

'''
generate label in VOC format
'''
def gen_label(class_name):
    val = []
    train = []
    val_txt = osp.join(main_path, class_name + 'val.txt')
    train_txt = osp.join(main_path, class_name+ 'train.txt')

    i = 0
    for file in os.listdir(osp.join(img_path, class_name)):
        i+=1
        if i % 5 == 0:
            val.append(file)
        else:
            train.append(file)
    print(i)
    print(len(val))
    print(len(train))

    # write validation file
    val_open=open(val_txt, 'w')
    for line in val:
        val_open.write(line + '\n')
    val_open.close()
    # write train file
    train_open=open(train_txt, 'w')
    for line in train:
        train_open.write(line + '\n')
    train_open.close()


if __name__ == "__main__":
    # gen xml file
    txt2xml(txt_path)
    # gen label file
    train = open('train.txt','w')
    val = open('val.txt','w')

    for file in os.listdir(main_path):
        file = osp.join(main_path, file)
        if 'val' in file:
            print(file)
            for line in open(file):
                val.writelines(line)
        if 'train' in file:
            print(file)
            for line in open(file):
                train.writelines(line)

    val.close()
    train.close()




