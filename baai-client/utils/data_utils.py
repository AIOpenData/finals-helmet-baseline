import glob
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from loguru import logger
from PIL import Image
from utils.tool_utils import horisontal_flip, load_classes, load_json, parse_data_config
from torch.utils.data import Dataset


def pad_to_square(image, pad_value):
    c, h, w = image.shape
    dim_diff = np.abs(h - w)

    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    # add padding
    image = F.pad(image, pad, "constant", value=pad_value)

    return image, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, image_folder_path: str, image_size: int):
        self.image_files = sorted(glob.glob("%s/*.*" % image_folder_path))
        self.image_size = image_size

    def __getitem__(self, index):
        image_path = self.image_files[index % len(self.image_files)]
        # extract image as PyTorch tensor
        image = transforms.ToTensor()(Image.open(image_path).convert('RGB'))
        # pad to square resolution
        image, _ = pad_to_square(image, 0)
        # resize
        image = resize(image, self.image_size)

        return image_path, image

    def __len__(self):
        return len(self.image_files)


class ListDataset(Dataset):
    def __init__(self, list_path: str, image_size: int, augment=True, multi_scale=True):
        # load train.json
        json_data = load_json(list_path)
        # folder for images
        self.image_file_root = list_path.replace("annotations", "images").split(".json")[0] + "/"
        # image information
        self.image_files = json_data['images']
        # label information
        self.label_files = json_data['annotations']
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multi_scale = multi_scale
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        image_path = self.image_file_root + self.image_files[index]['file_name']

        # extract image as PyTorch tensor
        image = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        # handle images with less than three channels
        if len(image.shape) != 3:
            image = image.unsqueeze(0)
            image = image.expand((3, image.shape[1:]))

        _, h, w = image.shape

        # pad to square resolution
        image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = image.shape

        image_id = self.image_files[index]['id']
        boxes = []
        for ann in self.label_files:
            if ann['image_id'] == image_id:
                cls_name = ann['category_id'] - 1  # start from 0
                x1 = np.float(ann['bbox'][0]) + pad[0]  # xmin
                x2 = np.float(ann['bbox'][0] + ann['bbox'][2]) + pad[1]  # xmax
                y1 = np.float(ann['bbox'][1]) + pad[2]
                y2 = np.float(ann['bbox'][1] + ann['bbox'][3]) + pad[3]
                x_center = ((x1 + x2) / 2) / padded_w
                y_center = ((y1 + y2) / 2) / padded_h
                w = ann['bbox'][2] / padded_w
                h = ann['bbox'][3] / padded_h
                boxes.append([cls_name, x_center, y_center, w, h])
        boxes = torch.Tensor(boxes)
        targets = torch.zeros((len(boxes), 6))
        if len(boxes) > 0:
            targets[:, 1:] = boxes

        # apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # remove empty placeholder targets
        if len(targets) > 0:
            targets = [boxes for boxes in targets if boxes is not None]

            # add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            targets = torch.cat(targets, 0)

        # selects new image size every tenth batch
        if self.multi_scale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets

    def __len__(self):
        return len(self.image_files)


def get_data(args=None):
    """ get the training dataset and testing dataset for the given client id by the customed data type """
    # metrics
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # get data configuration
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    test_path = data_config["test"]
    detect_path = data_config["detect"]

    # load test_image_info.json
    detect_json_data = load_json(detect_path)
    detect_image_folder_path = detect_path.replace("annotations", "images").split("_image_info.json")[0]

    # load names of classes
    class_names = load_classes(data_config["names"])

    train_dataset = ListDataset(list_path=train_path, augment=True, multi_scale=args.multi_scale_training,
                                image_size=args.image_size)
    valid_dataset = ListDataset(list_path=valid_path, augment=False, multi_scale=False, image_size=args.image_size)
    test_dataset = ListDataset(list_path=test_path, augment=False, multi_scale=False, image_size=args.image_size)
    detect_dataset = ImageFolder(image_folder_path=detect_image_folder_path, image_size=args.image_size)

    federated_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    federated_valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=False,
        collate_fn=valid_dataset.collate_fn,
    )
    federated_test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        pin_memory=False,
        collate_fn=test_dataset.collate_fn,
    )
    federated_detect_loader = torch.utils.data.DataLoader(
        detect_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
    )

    logger.info("client id: {}, federated train size: {}, federated valid size: {}, federated test size: {}".format(
        args.user_id, len(train_dataset), len(valid_dataset), len(test_dataset)))

    return federated_train_loader, federated_valid_loader, federated_test_loader, federated_detect_loader, len(
        train_dataset), class_names, metrics, detect_json_data
