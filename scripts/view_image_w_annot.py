import cv2
import matplotlib.pyplot as plt


color_palettes = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0)
]


def imshow(im):
    plt.imshow(im)
    plt.show()


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
        x_center = float(x_center) * w
        y_center = float(y_center) * h
        width = float(width) * w
        height = float(height) * h

        # if cls_label not in ['45']:
        #     continue

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=1)

    imshow(image)


if __name__ == '__main__':
    image_path = "/home/kancy/Desktop/okamura_dataset/yolo_train_data/train/images/005_w960h960_r-90_234.png"
    annot_path = "/home/kancy/Desktop/okamura_dataset/yolo_train_data/train/labels/005_w960h960_r-90_234.txt"

    annot_image(image_path, annot_path)