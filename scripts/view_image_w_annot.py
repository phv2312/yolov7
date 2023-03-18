import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

color_palettes = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (127, 240, 135)
]


def imshow(im):
    plt.imshow(im)
    plt.show()


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    import numpy as np
    import math
    import random

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    return img


def annot_image_polygon(image_path: str, annot_path: str):
    image = cv2.imread(image_path, 3)
    h, w = image.shape[:2]
    annots = open(annot_path, 'r').readlines()

    for annot in annots:
        infos = annot.split(' ')
        cls_label = infos[0]
        color = color_palettes[int(cls_label) % len(color_palettes)]

        points = list(map(float, infos[1:]))
        points = np.array(points).reshape((-1, 2))
        points = points * np.array([w, h]).reshape((1, 2))

        cv2.drawContours(image, [points[:, None, :].astype(int)], -1, color=color, thickness=2)

    imshow(image)


def annot_image(image_path: str, annot_path: str):
    image = cv2.imread(image_path, 3)
    h, w = image.shape[:2]
    annots = open(annot_path, 'r').readlines()

    for annot in annots:
        infos = annot.split(' ')
        cls_label = infos[0]
        print (cls_label)
        color = color_palettes[int(cls_label) % len(color_palettes)]

        x_center, y_center, width, height = infos[1:]

        if float(width) > 0.9:
            raise Exception("error")

        x_center = float(x_center) * w
        y_center = float(y_center) * h
        width = float(width) * w
        height = float(height) * h

        # print (x_center, y_center, width, height)

        # if cls_label not in ['45']:
        #     continue

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)

    imshow(image)


if __name__ == '__main__':
    basename = '008_w960h960_r0_467'

    image_dir = "/home/kancy/Desktop/okamura_dataset/yolo_train_data_segm/images/train/"
    label_dir = "/home/kancy/Desktop/okamura_dataset/yolo_train_data_segm/labels/train/"

    for image_path in os.listdir(image_dir):
        basename = os.path.splitext(image_path)[0]
        print (basename)

        image_path = os.path.join(image_dir, f"{basename}.png")
        label_path = os.path.join(label_dir, f"{basename}.txt")

        annot_image_polygon(image_path, label_path)