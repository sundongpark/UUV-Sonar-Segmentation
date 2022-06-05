import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import sys
import os
import json
import xml.etree.ElementTree as ET
import re

# python .\convert.py ./train_ann/ ./train_masks/ ./train/via_region_data.json
# python .\convert.py ./val_ann/ ./val_masks/ ./val/via_region_data.json
# python .\convert.py ./test_ann/ ./test_masks/ ./test/via_region_data.json
START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"Background": 0, "Bottle": 1, "Can": 2, "Chain": 3, "Drink-carton": 4,
#                           "Hook": 5, "Propeller": 6, "Shampoo-bottle": 7, "Standing-bottle": 8,
#                           "Tire": 9, "Valve": 10, "Wall": 11}

def get_contour(filename, image_dir):
    # Mask Image에서 경계선을 찾는다.
    xy_list = []
    mask = cv.imread(image_dir + filename, cv.IMREAD_GRAYSCALE)
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c = c.squeeze(1)
        c = c.transpose().tolist()
        xy_list.append(c)
    return xy_list, len(contours)


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    #english = re.compile(r'[a-zA-Z]')
    number = re.compile(r'[^0-9]')

    try:
        #숫자가  있는지 확인, 숫자만 추출하여 id에 넣기
        if number.match(filename):
            filename = filename.replace("\\", "/")[21:]
            filename = re.sub(r'[^0-9]', '', filename)
            return int(filename)
    except:
        print("error")


def get_categories(xml_files):
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, image_dir, json_file):
    # Convert annotations
    # VGG Image Annotator saves each image in the form:
    # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #   'regions': {
    #       '0': {
    #           'region_attributes': {},
    #           'shape_attributes': {
    #               'all_points_x': [...],
    #               'all_points_y': [...],
    #               'name': 'polygon'}},
    #       ... more regions ...
    #   },
    #   'size': 100202
    # }
    # We mostly care about the x and y coordinates of each region
#    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    json_dict = dict()
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        image_id = get_filename_as_int(filename)
        mask_list, l = get_contour(filename, image_dir)
        size = get_and_check(root, "size", 1)
        file_size = os.path.getsize("./Images/" + filename)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        json_dict[filename + str(file_size)] = {
                            "fileref": "",
                            "size": file_size,
                            "filename": filename,
                            "base64_img_data": "",
                            "file_attributes": {},
                            "regions": {}
        }
        object_i = 0
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "x", 1).text) - 1
            ymin = int(get_and_check(bndbox, "y", 1).text) - 1
            o_width = int(get_and_check(bndbox, "w", 1).text)
            o_height = int(get_and_check(bndbox, "h", 1).text)
            seg = mask_list
            for s in mask_list: # 이미지에 mask가 여러개이면
                if xmin <= (min(s[0]) + max(s[0]))/2 <= xmin + o_width and ymin <= (min(s[1]) + max(s[1]))/2 <= ymin + o_height: # bounding box 안에 있으면
                    seg = s     # 해당 mask
            seg = np.array(seg).squeeze().tolist()
            json_dict[filename + str(file_size)]["regions"][str(object_i)] = {"shape_attributes": {"name":"polygon", "all_points_x": seg[0], "all_points_y": seg[1]}, "region_attributes":{"name": category}}
            object_i += 1
            bnd_id = bnd_id + 1

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Mask to COCO format."
    )
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("image_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    args = parser.parse_args()
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, args.image_dir, args.json_file)
    print("Success: {}".format(args.json_file))
