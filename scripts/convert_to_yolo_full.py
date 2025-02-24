import glob
import os
import cv2
import json
import numpy as np
import math


LABELS_ACCEPTED = [
    'closed_workbooth', #6
    'conference_desk', #2
    'chair', #1
    'sofa_desk', #5
    'conference_sofa', #3
    'desk', #2
    'sofa', #3
    'cabinet' #4
]

LABELS_MAPPER = {
    'conference_desk': 'desk',
    'conference_sofa': 'sofa'
}


def get_shape(json_data):
    shapes = json_data['shapes']
    chair_results = [] # 1
    desk_results = [] # 2
    sofa_results = []  # 3
    cabinet_results = [] # 4
    sofa_desk_results = [] # 5
    closed_workbooth_results = [] # 6

    for shape in shapes:
        label = shape['label']
        if label not in LABELS_ACCEPTED:
            continue

        points = np.array(shape['points'], dtype=np.float32)
        if shape['shape_type'] == 'circle':
            center = (points[0][0], points[0][1])
            radius = int(
                math.sqrt(
                    (points[0][0] - points[1][0]) ** 2
                    + (points[0][1] - points[1][1]) ** 2
                )
            )
            x_min, y_min = center[0] - radius, center[1] - radius
            width = 2 * radius
            height = 2 * radius
        else:
            x_min, y_min, width, height = cv2.boundingRect(points[:, None, :])

        if label in ['chair']:
            chair_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['desk', 'conference_desk']:
            desk_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['sofa']:
            sofa_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['cabinet']:
            cabinet_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['sofa_desk']:
            sofa_desk_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['closed_workbooth']:
            closed_workbooth_results += [(x_min, y_min, x_min + width, y_min + height)]
    return (
        np.array(chair_results).astype(np.float32),
        np.array(desk_results).astype(np.float32),
        np.array(sofa_results).astype(np.float32),
        np.array(cabinet_results).astype(np.float32),
        np.array(sofa_desk_results).astype(np.float32),
        np.array(closed_workbooth_results).astype(np.float32)
    )


if __name__ == '__main__':
    image_dir = "/home/kancy/Desktop/okamura_dataset/v0302_w_synthesis_blur_external/all"
    label_dir = "/home/kancy/Desktop/okamura_dataset/v0302_w_synthesis_blur_external/all"
    split_json_path = "/home/kancy/Desktop/okamura_dataset/split.json"
    output_dir = "/home/kancy/Desktop/okamura_dataset/yolo_train_data_big"

    # read split data & make dirs if need
    split_data = json.load(open(split_json_path, 'r'))

    for dir_name in ['train', 'val']:
        output_image_dir = os.path.join(output_dir, 'images', dir_name)
        output_label_dir = os.path.join(output_dir, 'labels', dir_name)

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    count = 0
    rotate_degree = 0
    for json_path in glob.glob(os.path.join(label_dir, "*.json")):
        if not json_path.endswith('.json'):
            continue

        print(f'processing: {json_path}, rotate degree: {rotate_degree} ...')

        basename = os.path.splitext(os.path.basename(json_path))[0]
        image_path = os.path.join(image_dir, f"{basename}.png")
        output_path = f"{basename}.txt"

        print('json_path:', json_path)
        print('image_path:', image_path)
        print('output_path:', output_path)

        image = cv2.imread(image_path, 3)
        image_h, image_w = image.shape[:2]
        json_data = json.load(open(json_path, 'r'))
        chairs, desks, sofas, cabinets, sofa_desks, closed_workbooths = get_shape(json_data)

        if basename in split_data['test']:
            # val
            if rotate_degree != 0:
                continue

            output_image_dir_ = os.path.join(output_dir, 'images', 'val')
            output_label_dir_ = os.path.join(output_dir, 'labels', 'val')
        else:
            # train
            output_image_dir_ = os.path.join(output_dir, 'images', 'train')
            output_label_dir_ = os.path.join(output_dir, 'labels', 'train')

        # write image
        basename_annot = f"{basename}_r{rotate_degree}_{count}"
        cv2.imwrite(
            os.path.join(output_image_dir_, f"{basename_annot}.png"),
            image
        )

        # write label
        writer = open(os.path.join(output_label_dir_, f"{basename_annot}.txt"), 'w')

        # write
        for ci, objects_ in enumerate([chairs, desks, sofas, cabinets, sofa_desks, closed_workbooths]):
            for object_ in objects_:
                x_min, y_min, x_max, y_max = object_
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = (x_max - x_min + 1)
                height = (y_max - y_min + 1)

                assert x_min >= 0
                assert y_min >= 0
                assert width > 0
                assert height > 0

                # normalize
                x_center = x_center / image_w
                y_center = y_center / image_h
                width = width / image_w
                height = height / image_h

                line_str = map(str, [ci, x_center, y_center, width, height])
                writer.write(" ".join(line_str))
                writer.write("\n")
