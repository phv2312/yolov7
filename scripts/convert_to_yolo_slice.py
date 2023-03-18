import glob
import json
import math
import os
import cv2

from typing import Dict, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import imutils
import random


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


def imshow(im):
    plt.imshow(im)
    plt.show()

@logger.catch
def random_gen_long_line(
    image: np.ndarray,
    desks: np.ndarray,
    min_lines: int = 3,
    max_lines: int = 8,
    black_value: int = 50
) -> np.ndarray:
    h, w = image.shape[:2]

    if len(desks) < 1:
        return image

    horizontal_lines = random.randint(min_lines, max_lines)
    vertical_lines = random.randint(min_lines, max_lines)

    box_range_x = (np.min(desks[:, 0]), np.max(desks[:, 2]))
    box_range_y = (np.min(desks[:, 1]), np.max(desks[:, 3]))

    for i in range(0, horizontal_lines):
        random_x = random.randint(int(box_range_x[0]), int(box_range_x[1]) - 5)
        y_from = random.randint(0, int(0.1 * h))
        y_to = random.randint(int(0.8 * h), h - 1)

        image[y_from: y_to, random_x, :] = random.randint(0, black_value)
        image[y_from: y_to, random_x + 1, :] = random.randint(0, black_value)
    for j in range(0, vertical_lines):
        random_y = random.randint(int(box_range_y[0]), int(box_range_y[1]) - 5)
        x_from = random.randint(0, int(0.1 * w))
        x_to = random.randint(int(0.8 * w), w - 1)

        image[random_y, x_from: x_to, :] = random.randint(0, black_value)
        image[random_y + 1, x_from: x_to, :] = random.randint(0, black_value)

    return image


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


def get_shape(json_data, extract_polygon: bool = False):
    shapes = json_data['shapes']

    chair_results = [] #1
    desk_results = [] #2
    sofa_results = [] #3
    cabinet_results = [] #4
    sofa_desk_results = [] #5
    closed_workbooth_results = [] #6

    for shape in shapes:
        label = shape['label']
        if label not in LABELS_ACCEPTED:
            continue

        if shape['shape_type'] == 'rectangle':
            points = np.array(shape['points'], dtype=np.float32)
            x_min, y_min, width, height = cv2.boundingRect(points[:, None, :])

            points = np.array([
                [x_min, y_min],
                [x_min + width, y_min],
                [x_min + width, y_min + height],
                [x_min, y_min + height]
            ])

        elif shape['shape_type'] == 'circle':
            points = np.array(shape['points'])
            center = (int(points[0][0]), int(points[0][1]))
            radius = int(
                math.sqrt(
                    (points[0][0] - points[1][0]) ** 2
                    + (points[0][1] - points[1][1]) ** 2
                )
            )

            n_sample = 15
            points = []
            for i in range(n_sample):
                x = center[0] + radius * math.cos(i * 2 * math.pi / n_sample)
                y = center[1] + radius * math.sin(i * 2 * math.pi / n_sample)
                points += [(x, y)]
            points = np.array(points) # (n, 2)

        elif shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.float32) # (n, 2)

        else:
            raise Exception("Not know!")

        # group labels
        if extract_polygon is False:
            x_min, y_min, width, height = cv2.boundingRect(points[:, None, :])
            x_max = x_min + width
            y_max = y_min + height

            object_annot = [(x_min, y_min, x_max, y_max)]
        else:
            object_annot = [
                np.array([tuple(point) for point in points.tolist()], dtype=float)
            ]

        if label in ['chair']:#1
            chair_results += object_annot
        elif label in ['desk', 'conference_desk']:#2
            desk_results += object_annot
        elif label in ['sofa', 'conference_sofa']:#3
            sofa_results += object_annot
        elif label in ['cabinet']: #4
            cabinet_results += object_annot
        elif label in ['sofa_desk']: #5
            sofa_desk_results += object_annot
        elif label in['closed_workbooth']:
            closed_workbooth_results += object_annot

    if extract_polygon is False:
        return (
            np.array(chair_results).astype(np.float32),
            np.array(desk_results).astype(np.float32),
            np.array(sofa_results).astype(np.float32),
            np.array(cabinet_results).astype(np.float32),
            np.array(sofa_desk_results).astype(np.float32),
            np.array(closed_workbooth_results).astype(np.float32)
        )
    else:
        return (
            chair_results,
            desk_results,
            sofa_results,
            cabinet_results,
            sofa_desk_results,
            closed_workbooth_results
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


def get_shapes_inside(shapes: np.ndarray, rect: Tuple, center: Tuple, rotate_degree: int, origin_shape: Tuple):
    if len(shapes) < 1:
        return []
    origin_h, origin_w = origin_shape[:2]
    center = np.array(center, dtype=float)

    # points -> bounding boxes
    bounding_boxes = []
    for i, shape in enumerate(shapes):
        x_min, y_min, width, height = cv2.boundingRect(shape[:, None, :].astype(int))
        x_max = x_min + width
        y_max = y_min + height

        bounding_boxes += [(x_min, y_min, x_max, y_max)]
    bounding_boxes = np.array(bounding_boxes)
    # end <<

    x_min, y_min, x_max, y_max = rect

    xx_min = np.maximum(bounding_boxes[:, 0], x_min)
    yy_min = np.maximum(bounding_boxes[:, 1], y_min)
    xx_max = np.minimum(bounding_boxes[:, 2], x_max)
    yy_max = np.minimum(bounding_boxes[:, 3], y_max)

    w = np.maximum(0.0, xx_max - xx_min)
    h = np.maximum(0.0, yy_max - yy_min)
    inter = w * h
    shape_area = (bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1) * (bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1)

    ious = inter / shape_area
    valid_ids = np.where(ious > 0.5)[0]

    mask1 = np.zeros(shape=(origin_h, origin_w), dtype=np.uint8)
    mask1[y_min: y_max, x_min: x_max] = 1
    if len(valid_ids) > 0:
        new_shapes = [shapes[i] for i in valid_ids]

        # # rotate
        # if rotate_degree != 0:
        #     new_shapes = rotate_bbox(new_shapes, center, rotate_degree)
        #
        #     new_shapes_ = new_shapes.copy()
        #     new_shapes_[:, 0] = np.minimum(new_shapes[:, 0], new_shapes[:, 2]) # x_min
        #     new_shapes_[:, 1] = np.minimum(new_shapes[:, 1], new_shapes[:, 3]) # y_min
        #     new_shapes_[:, 2] = np.maximum(new_shapes[:, 0], new_shapes[:, 2]) # x_max
        #     new_shapes_[:, 3] = np.maximum(new_shapes[:, 1], new_shapes[:, 3]) # y_max
        #
        #     new_shapes = new_shapes_

        # clip
        for i in range(len(new_shapes)):
            # print (f'{i},iou:', ious[valid_ids[i]])
            shape_ = new_shapes[i].astype(int)

            # check whether we need to perform contour intersection
            do_intersect = True
            x_min_, y_min_, width_, height_ = cv2.boundingRect(shape_[:, None, :])
            if (x_min <= x_min_ <= x_min_ + width_ <= x_max) and (y_min <= y_min_ <= y_min_ + height_ <= y_max):
                do_intersect = False

            if do_intersect:
                mask2 = np.zeros_like(mask1)
                cv2.drawContours(mask2, [shape_[:, None, :]], -1, 1, thickness=-1)

                # find the intersection between two contours
                intersection_mask = mask1 & mask2
                # plt.imshow(intersection_mask); plt.show()
                contours = cv2.findContours(intersection_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

                try:
                    contour = sorted(contours, key=lambda elem: cv2.contourArea(elem))[::-1][0]
                except IndexError:
                    plt.imshow(mask2); plt.show()
                    plt.imshow(mask1); plt.show()
                    plt.imshow(intersection_mask); plt.show()

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
                new_shapes[i] = approx[:, 0, :] - np.array([x_min, y_min]).reshape(1, 2)
                # end <<
            else:
                new_shapes[i] = np.array([s - np.array([x_min, y_min]) for s in shape_])

            new_shapes[i][:, 0] = np.clip(new_shapes[i][:, 0], 0., max(y_max - y_min + 1, x_max - x_min + 1))
            new_shapes[i][:, 1] = np.clip(new_shapes[i][:, 1], 0., max(y_max - y_min + 1, x_max - x_min + 1))
            new_shapes[i] = new_shapes[i].astype(int)

            # get unique, but keep order
            uniques = []
            new_shapes_unique_keep_ord = []
            for elem in new_shapes[i]:
                k = tuple(elem)

                if k not in uniques:
                    new_shapes_unique_keep_ord += [k]
                    uniques += [k]
            # <<
            new_shapes[i] = np.array(new_shapes_unique_keep_ord)

        # new_shapes[:, 0] = np.clip(new_shapes[:, 0], 0., max(y_max - y_min + 1, x_max - x_min + 1))
        # new_shapes[:, 1] = np.clip(new_shapes[:, 1], 0., max(y_max - y_min + 1, x_max - x_min + 1))
        # new_shapes[:, 2] = np.clip(new_shapes[:, 2], 0., max(y_max - y_min + 1, x_max - x_min + 1))
        # new_shapes[:, 3] = np.clip(new_shapes[:, 3], 0., max(y_max - y_min + 1, x_max - x_min + 1))

        # get unique
        # new_shapes = np.unique(new_shapes.astype(int), axis=0)

        return new_shapes

    return []


if __name__ == '__main__':
    image_dir = "/home/kancy/Desktop/okamura_dataset/v0302_w_synthesis_blur_external/all"
    label_dir = "/home/kancy/Desktop/okamura_dataset/v0302_w_synthesis_blur_external/all"
    split_json_path = "/home/kancy/Desktop/okamura_dataset/split.json"
    output_dir = "/home/kancy/Desktop/okamura_dataset/yolo_train_data_segm"

    slide = (480, 480)
    window = (1280, 1280)
    output_size = (800, 800)

    use_polygon = True

    # read split data & make dirs if need
    split_data = json.load(open(split_json_path, 'r'))

    for dir_name in ['train', 'val']:
        output_image_dir = os.path.join(output_dir, 'images', dir_name)
        output_label_dir = os.path.join(output_dir, 'labels', dir_name)

        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    for rotate_degree in [0]:
        count = 0
        for json_path in glob.glob(os.path.join(label_dir, '*.json')):
            if not json_path.endswith('.json'):
                continue

            print(f'processing: {json_path}, rotate degree: {rotate_degree} ...')

            basename = os.path.splitext(os.path.basename(json_path))[0]

            # if basename not in ['034']:
            #     continue

            image_path = os.path.join(image_dir, f"{basename}.png")
            output_path = f"{basename}.txt"

            print('json_path:', json_path)
            print('image_path:', image_path)
            print('output_path:', output_path)

            image = cv2.imread(image_path, 3)
            image_h, image_w = image.shape[:2]
            json_data = json.load(open(json_path, 'r'))
            chairs, desks, sofas, cabinets, sofa_desks, closed_workbooths = get_shape(json_data, extract_polygon=use_polygon)

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

            for y in range(0, image_h, slide[1]):
                for x in range(0, image_w, slide[0]):
                    print (x, y)
                    # x = 960
                    # y = 0

                    # get slice
                    image_ = image[y:y + window[1], x:x + window[0], :]
                    h_, w_ = image_.shape[:2]

                    # create square input
                    if window[1] > h_:
                        image_ = cv2.copyMakeBorder(image_, 0, window[1] - h_, 0, 0, cv2.BORDER_CONSTANT, value=0)

                    if window[0] > w_:
                        image_ = cv2.copyMakeBorder(image_, 0, 0, 0, window[0] - w_, cv2.BORDER_CONSTANT, value=0)

                    h_, w_ = image_.shape[:2]
                    center_ = (w_ / 2., h_ / 2.)

                    image_ = imutils.rotate(image_, angle=-rotate_degree, center=center_)

                    print('desk')
                    desks_ = get_shapes_inside(
                        desks, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    print ('sofa')
                    sofas_ = get_shapes_inside(
                        sofas, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    print('chair')
                    chairs_ = get_shapes_inside(
                        chairs, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    print ('cabinet')
                    cabinets_ = get_shapes_inside(
                        cabinets, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    print ('sofa_desk')
                    sofa_desks_ = get_shapes_inside(
                        sofa_desks, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    print ('closed_workbooth')
                    closed_workbooths_ = get_shapes_inside(
                        closed_workbooths, (x, y, x + window[0], y + window[1]), center_, rotate_degree, (image_h, image_w)
                    )

                    # #
                    # if (np.random.uniform(0, 1.) < 0.15) and (basename not in split_data['test']):
                    #     image_ = random_gen_long_line(image_, desks_)

                    if False:
                        # try to visualize
                        image_vis_ = image_.copy()

                        #
                        # for x_min, y_min, x_max, y_max in chairs_:
                        #     cv2.rectangle(image_vis_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=1)
                        # for x_min, y_min, x_max, y_max in desks_:
                        #     cv2.rectangle(image_vis_, (x_min, y_min), (x_max, y_max), color=(255, 0, 255), thickness=1)

                        #
                        cv2.drawContours(image_vis_, chairs_, -1, color=(255, 0, 0), thickness=2)
                        cv2.drawContours(image_vis_, desks_, -1, color=(255, 0, 255), thickness=2)
                        cv2.drawContours(image_vis_, sofas_, -1, color=(0, 255, 0), thickness=2)

                        imshow(image_vis_)
                        continue
                        # end <<<

                    is_skip = False
                    if len(desks_) < 1 and len(chairs_) < 1 and len(cabinets_) < 1 and \
                        len(sofas_) < 1 and len(sofa_desks_) < 1 and len(closed_workbooths_) < 1:
                        is_skip = True

                        if np.random.uniform(0, 1.) < 0.15:
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
                        cv2.resize(image_, output_size, interpolation=cv2.INTER_CUBIC)
                    )

                    # write label
                    writer = open(os.path.join(output_label_dir_, f"{basename_annot}.txt"), 'w')
                    for ci, objects_ in enumerate([chairs_, desks_, sofas_, cabinets_, sofa_desks_, closed_workbooths_]):
                        for object_ in objects_:
                            if use_polygon:
                                norm_object_ = object_ / np.array([image_.shape[0] + 1, image_.shape[1] + 1]).reshape(1, 2)
                                for coord in norm_object_:
                                    coord_x, coord_y = coord

                                    assert 0 <= coord_x <= 1, f'{coord_x}'
                                    assert 0 <= coord_y <= 1, f'{coord_y}'

                                line_str = list(map(str, [ci] + norm_object_.reshape((-1,)).tolist()))
                            else:
                                x_min, y_min, width, height = cv2.boundingRect(object_[:, None, :])
                                x_center = x_min + width / 2.
                                y_center = y_min + width / 2

                                # normalize
                                x_center = x_center / image_.shape[1]
                                y_center = y_center / image_.shape[0]
                                width = width / image_.shape[1]
                                height = height / image_.shape[0]

                                assert 0 <= x_center < 1, f"{x_center}"
                                assert 0 <= y_center < 1, f"{y_center}"
                                assert 0 < width < 1., f"{width}"
                                assert 0 < height < 1., f"{height}"

                                line_str = map(str, [ci, x_center, y_center, width, height])

                            writer.write(" ".join(line_str))
                            writer.write("\n")

                    count += 1

