import glob
import json
import math
import os
import cv2

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import imutils


LABELS_ACCEPTED = ['chair', 'desk', 'cabinet', 'conference_chair', 'conference_desk']


def imshow(im):
    plt.imshow(im)
    plt.show()



def _label2mask(label: Dict, image_size: Tuple) -> np.ndarray:
    w, h = image_size
    n_cls = 2
    mask = np.zeros(shape=(n_cls, h, w), dtype=np.uint8)
    shapes = label['shapes']
    for shape in shapes:
        label = shape['label']
        if label not in LABELS_ACCEPTED:
            continue
        points = np.array(shape['points'], dtype=np.float32)
        x_min, y_min, width, height = cv2.boundingRect(points[:, None, :])

        if label == 'chair':
            mask[0, y_min: y_min + height, x_min: x_min + width] = 1.
        elif label == 'desk':
            mask[1, y_min: y_min + height, x_min: x_min + width] = 1.
        else:
            raise NotImplementedError("not implement")
    return mask


def get_shape(json_data):
    shapes = json_data['shapes']
    chair_results = []
    desk_results = []
    cabinet_results = []
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

        if label in ['chair', 'conference_chair']:
            chair_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label in ['desk', 'conference_desk']:
            desk_results += [(x_min, y_min, x_min + width, y_min + height)]
        elif label == 'cabinet':
            cabinet_results += [(x_min, y_min, x_min + width, y_min + height)]

    return (
        np.array(chair_results).astype(np.float32),
        np.array(desk_results).astype(np.float32),
        np.array(cabinet_results).astype(np.float32)
    )


def rotate_bbox(bboxes_np: np.ndarray, origin: np.ndarray, rotate_degree: int) -> np.ndarray:
    rotate_radian = math.radians(rotate_degree)
    rotate_mat = np.array([
        [math.cos(rotate_radian), -math.sin(rotate_radian)],
        [math.sin(rotate_radian), math.cos(rotate_radian)]
    ])

    n_bbox = bboxes_np.shape[0]
    origin = origin.reshape((1, 2))
    points = bboxes_np.reshape((n_bbox * 2, 2))
    points_normalized = points - origin

    points_transformed = (rotate_mat @ points_normalized.T).T

    return (points_transformed + origin).reshape((-1, 4))


def get_shapes_inside(shapes: np.ndarray, rect: Tuple, center: Tuple, rotate_degree: int ):
    if len(shapes) < 1:
        return []

    center = np.array(center, dtype=float)

    x_min, y_min, x_max, y_max = rect

    xx_min = np.maximum(shapes[:, 0], x_min)
    yy_min = np.maximum(shapes[:, 1], y_min)
    xx_max = np.minimum(shapes[:, 2], x_max)
    yy_max = np.minimum(shapes[:, 3], y_max)

    w = np.maximum(0.0, xx_max - xx_min + 1)
    h = np.maximum(0.0, yy_max - yy_min + 1)
    inter = w * h
    shape_area = (shapes[:,3] - shapes[:, 1] + 1) * (shapes[:, 2] - shapes[:, 0] + 1)

    ious = inter / shape_area
    valid_ids = np.where(ious > 0.2)[0]

    if len(valid_ids) > 0:
        new_shapes = shapes[valid_ids].copy()

        new_shapes[:, 0] -= x_min
        new_shapes[:, 1] -= y_min
        new_shapes[:, 2] -= x_min
        new_shapes[:, 3] -= y_min

        # rotate
        if rotate_degree != 0:
            new_shapes = rotate_bbox(new_shapes, center, rotate_degree)

            new_shapes_ = new_shapes.copy()
            new_shapes_[:, 0] = np.minimum(new_shapes[:, 0], new_shapes[:, 2]) # x_min
            new_shapes_[:, 1] = np.minimum(new_shapes[:, 1], new_shapes[:, 3]) # y_min
            new_shapes_[:, 2] = np.maximum(new_shapes[:, 0], new_shapes[:, 2]) # x_max
            new_shapes_[:, 3] = np.maximum(new_shapes[:, 1], new_shapes[:, 3]) # y_max

            new_shapes = new_shapes_

        # get unique
        new_shapes = np.unique(new_shapes, axis=0)

        # clip
        new_shapes[:, 0] = np.clip(new_shapes[:, 0], 0., x_max - x_min)
        new_shapes[:, 1] = np.clip(new_shapes[:, 1], 0., y_max - y_min)
        new_shapes[:, 2] = np.clip(new_shapes[:, 2], 0., x_max - x_min)
        new_shapes[:, 3] = np.clip(new_shapes[:, 3], 0., y_max - y_min)

        return new_shapes.astype(int)

    return []

if __name__ == '__main__':
    image_dir = "/home/kancy/Desktop/okamura_dataset/v021/all"
    label_dir = "/home/kancy/Desktop/okamura_dataset/v021/all"
    split_json_path = "/home/kancy/Desktop/okamura_dataset/split.json"
    output_dir = "/home/kancy/Desktop/okamura_dataset/yolo_train_data"

    slide = (960, 960)
    window = (1280, 1280)

    # read split data & make dirs if need
    split_data = json.load(open(split_json_path, 'r'))

    for dir_name in ['train', 'val']:
        dir_abs_path = os.path.join(output_dir, dir_name)
        output_image_dir = os.path.join(dir_abs_path, 'images')
        output_label_dir = os.path.join(dir_abs_path, 'labels')

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    for rotate_degree in [0]:
        count = 0
        for json_path in glob.glob(os.path.join(label_dir, '*.json')):

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
            chairs, desks, cabinets = get_shape(json_data)

            if basename in split_data['train']:
                # train
                output_image_dir_ = os.path.join(output_dir, 'train', 'images')
                output_label_dir_ = os.path.join(output_dir, 'train', 'labels')
            else:
                # val
                if rotate_degree != 0:
                    continue

                output_image_dir_ = os.path.join(output_dir, 'val', 'images')
                output_label_dir_ = os.path.join(output_dir, 'val', 'labels')

            for y in range(0, image_h, slide[1]):
                for x in range(0, image_w, slide[0]):
                    image_ = image[y:y + window[1], x:x + window[0], :]
                    h_, w_ = image_.shape[:2]
                    center_ = (w_ / 2., h_ / 2.)

                    image_ = imutils.rotate(image_.copy(), angle=-rotate_degree, center=center_)

                    chairs_ = get_shapes_inside(
                        chairs, (x, y, x + window[0], y + window[1]), center_, rotate_degree
                    )
                    desks_ = get_shapes_inside(
                        desks, (x, y, x + window[0], y + window[1]), center_, rotate_degree
                    )
                    cabinets_ = get_shapes_inside(
                        cabinets, (x, y, x + window[0], y + window[1]), center_, rotate_degree
                    )

                    if False:
                        # try to visualize
                        image_vis_ = image_.copy()
                        # for x_min, y_min, x_max, y_max in chairs_:
                        #     cv2.rectangle(image_vis_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=1)

                        for x_min, y_min, x_max, y_max in cabinets_:
                            cv2.rectangle(image_vis_, (x_min, y_min), (x_max, y_max), color=(255, 0, 255), thickness=2)

                        imshow(image_vis_)
                        continue
                        # end <<<

                    is_skip = False
                    if len(desks_) < 1 and len(chairs_) < 1 and len(cabinets_) < 1:
                        is_skip = True

                        if np.random.uniform(0, 1.) < 0.1:
                            is_skip = False
                            print ('Create empty data')
                    else:
                        is_skip = False

                    if is_skip:
                        continue

                    # write image
                    basename_annot = f"{basename}_w{slide[0]}h{slide[1]}_r{rotate_degree}_{count}"
                    cv2.imwrite(
                        os.path.join(output_image_dir_, f"{basename_annot}.png"),
                        image_
                    )

                    # write label
                    writer = open(os.path.join(output_label_dir_, f"{basename_annot}.txt"), 'w')
                    for ci, objects_ in enumerate([chairs_, desks_, cabinets_]):
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
                            x_center = x_center / image_.shape[1]
                            y_center = y_center / image_.shape[0]
                            width = width / image_.shape[1]
                            height = height / image_.shape[0]

                            line_str = map(str, [ci, x_center, y_center, width, height])
                            writer.write(" ".join(line_str))
                            writer.write("\n")

                    count += 1






