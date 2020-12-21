from __future__ import division
import copy
import numpy as np
import os
import pickle
import torch
import tqdm

from config.options_conf import args_parser
from loguru import logger
from module.models_module import Darknet
from PIL import Image
from terminaltables import AsciiTable
from torch.autograd import Variable
from utils.common_utils import CommonUtils
from utils.data_utils import get_data
from utils.result_utils import ResultUtils
from utils.tool_utils import ap_per_class, get_batch_statistics, namelist, non_max_suppression, rescale_boxes, \
    set_random_seed, timer, weights_init_normal, xywh2xyxy

# arguments parsing
args = args_parser()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


class Client(object):
    """ the client class is responsible for federated training on the local training dataset """
    # set random seed for list, numpy, CPU, current GPU and all the GPUs
    set_random_seed(args)

    # create folders for saving model and log information
    args.model_folder_path = os.path.join("./save")
    args.log_folder_path = os.path.join("./log")

    if not os.path.exists(args.model_folder_path):
        os.makedirs(args.model_folder_path)
    if not os.path.exists(args.log_folder_path):
        os.makedirs(args.log_folder_path)

    # add device, model and log file path arguments
    args.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu != -1) else "cpu")

    args.model_file_path = os.path.join(args.model_folder_path,
                                        "D_{}_M_{}_SE_{}_CE_{}_ID_{}.pkl".format(args.dataset, args.model,
                                                                                 args.server_epoch, args.client_epoch,
                                                                                 args.user_id))
    args.log_file_path = os.path.join(args.log_folder_path,
                                      "D_{}_M_{}_SE_{}_CE_{}_ID_{}.log".format(args.dataset, args.model,
                                                                               args.server_epoch, args.client_epoch,
                                                                               args.user_id))
    args.submission_file_path = os.path.join(args.log_folder_path,
                                             "D_{}_M_{}_SE_{}_CE_{}_ID_{}.json".format(args.dataset, args.model,
                                                                                       args.server_epoch,
                                                                                       args.client_epoch,
                                                                                       args.user_id))

    # initialize log output configuration
    logger.add(args.log_file_path)

    # initiate model and load pretrained model weights
    model = Darknet(config_path=args.model_def, image_size=args.image_size).to(args.device)
    model.apply(weights_init_normal)

    # get the training, evaluation and testing dataloader
    federated_train_loader, federated_valid_loader, federated_test_loader, federated_detect_loader, \
    federated_train_size, class_names, metrics, detect_json_data = get_data(args=args)
    detect_namelist = namelist(detect_json_data)

    @classmethod
    def get_model(cls, model_params: object):
        """ get a deep copy model (do DataParallel if there are multiple GPUs) """
        model = copy.deepcopy(cls.model)
        model.load_state_dict(model_params)
        # do training, evaluating and testing on multiple GPUs if there are multiple GPUs
        model = torch.nn.DataParallel(model, device_ids=[0]).to(args.device) if (
                torch.cuda.device_count() > 1) else model.to(args.device)
        return model

    @classmethod
    def get_model_params(cls, model: object):
        """ get model parameters (from module if there are multiple GPUs) """
        # reference to: https://www.zhihu.com/question/67726969/answer/511220696
        model_params = model.module.state_dict() if (torch.cuda.device_count() > 1) else model.state_dict()
        return model_params

    @classmethod
    def get_federated_train_size(cls):
        """ get the training dataset size for current client """
        return ResultUtils.success(data={"federated_train_size": cls.federated_train_size})

    @classmethod
    def train(cls, server_model_params=None, epoch=None):
        """ the current client does federated training on the local training dataset
            based on the current server epoch and the latest server model parameters
        """
        logger.info("start user_id {} training for epoch {}!".format(args.user_id, epoch))
        model = cls.get_model(model_params=server_model_params)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        epo_avg_loss = 0.0
        for epo in range(1, args.client_epoch + 1):
            with timer("federated train for epoch {}/{}, idx {}/{}, epo {}/{}".format(epoch, args.server_epoch,
                                                                                      args.user_id, args.user_num, epo,
                                                                                      args.client_epoch), logger):
                avg_loss = 0.0
                for batch_idx, (_, images, labels) in enumerate(cls.federated_train_loader):
                    done_batches = len(cls.federated_train_loader) * epoch + batch_idx

                    images = Variable(images.to(args.device))
                    labels = Variable(labels.to(args.device), requires_grad=False)

                    loss, outputs = model(images, labels)
                    loss.backward()

                    if done_batches % args.gradient_accumulations:
                        # accumulates gradient before each step
                        optimizer.step()
                        optimizer.zero_grad()

                    if args.verbose and batch_idx % args.log_interval == 0:
                        log_str = "\nfederated train for [{}/{} ({:.0f}%)]\tloss: {:.6f}\n".format(
                            batch_idx * len(images), len(cls.federated_train_loader.dataset),
                            100. * batch_idx / len(cls.federated_train_loader), loss.item())
                        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                        # log metrics at each YOLO layer
                        for i, metric in enumerate(cls.metrics):
                            formats = {m: "%.6f" for m in cls.metrics}
                            formats["grid_size"] = "%2d"
                            formats["cls_acc"] = "%.2f%%"
                            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                            metric_table += [[metric, *row_metrics]]

                        log_str += AsciiTable(metric_table).table

                        logger.info(log_str)

                        model.seen += images.size(0)

                    # update the average training loss for the client based on the batch
                    avg_loss += (loss.item() - avg_loss) / (batch_idx + 1)
                logger.info("epo {:3d}, average loss {:.3f}".format(epo, avg_loss))

                if epo % args.evaluation_interval == 0:
                    # evaluate the model on the validation set
                    client_model_params = cls.get_model_params(model=model)
                    cls().test(test_model_params=client_model_params, mode="valid")

                if epo % args.checkpoint_interval == 0:
                    CommonUtils.get_pickle_file_by_object_func(target=model.state_dict(),
                                                               write_file_path=args.model_file_path)

                # update the average training loss for the client
                epo_avg_loss += (avg_loss - epo_avg_loss) / epo
        logger.info("user id {:3d}, average loss {:.3f}".format(args.user_id, epo_avg_loss))

        model = model.to(torch.device("cpu"))
        client_model_params = cls.get_model_params(model=model)
        cls().test(test_model_params=client_model_params, mode="test")

        return ResultUtils.success(data=CommonUtils.get_pickle_bytes_by_object_func(
            {"client_model_params": client_model_params, "epo_avg_loss": epo_avg_loss}),
            media_type="application/octet-stream")

    @classmethod
    def test(cls, test_model_params=None, mode=None):
        """ get the valid / test results of the federated averaging model on the valid / test set """
        with timer("{} for user id {}".format("evaluate" if (mode == "valid") else "test", args.user_id), logger):
            model = cls.get_model(model_params=test_model_params)
            model.eval()

            label_list = []
            metric_list = []
            for idx, (_, images, labels) in enumerate(
                    tqdm.tqdm(cls.federated_valid_loader if (mode == "valid") else cls.federated_test_loader,
                              desc=mode)):
                images = Variable(images.to(args.device), requires_grad=False)

                # extract labels
                label_list += labels[:, 1].tolist()
                # rescale target
                labels[:, 2:] = xywh2xyxy(labels[:, 2:])
                labels[:, 2:] *= args.image_size

                with torch.no_grad():
                    outputs = model(images)
                    outputs = non_max_suppression(outputs, conf_threshold=args.conf_threshold,
                                                  nms_threshold=args.nms_threshold)

                metric_list += get_batch_statistics(outputs, labels, iou_threshold=args.iou_threshold)

            if len(metric_list) != 0:
                true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metric_list))]
                precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, label_list)
            else:
                logger.info("the metric list is empty!")

            if args.verbose and (len(metric_list) != 0):
                # logger.info class APs and mAP
                ap_table = [["Index", "Class name", "Precision", "Recall", "F1", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [
                        [c, cls.class_names[c], "%.5f" % precision[i], "%.5f" % recall[i], "%.5f" % f1[i],
                         "%.5f" % AP[i]]]
                ap_table += [[len(ap_class), "Total", "%.5f" % precision.mean(), "%.5f" % recall.mean(),
                              "%.5f" % f1.mean(), "%.5f" % AP.mean()]]
                logger.info("\n{}".format(AsciiTable(ap_table).table))
                logger.info(f"---- mAP {AP.mean()}")

            return ResultUtils.success(data=CommonUtils.get_pickle_bytes_by_object_func({"test": True}),
                                       media_type="application/octet-stream")

    @classmethod
    def detect(cls, detect_model_params: object):
        """ get the test results and save the json file of the federated averaging model on the test set """
        with timer("detect for user id {}".format(args.user_id), logger):
            model = cls.get_model(model_params=detect_model_params)
            model.eval()

            image_list = []  # store image paths
            image_detection_list = []  # store image detection for each image index

            for idx, (image_path_list, images) in enumerate(tqdm.tqdm(cls.federated_detect_loader, desc="detect")):
                images = Variable(images.to(args.device), requires_grad=False)

                with torch.no_grad():
                    outputs = model(images)
                    outputs = non_max_suppression(outputs, conf_threshold=args.conf_threshold,
                                                  nms_threshold=args.nms_threshold)

                # save images and detections
                image_list.extend(image_path_list)
                image_detection_list.extend(outputs)

            annotation_list = []
            for idx, (image_path, image_detection) in enumerate(zip(image_list, image_detection_list)):
                # find the image id corresponding to the image path
                image = np.array(Image.open(image_path))
                if image_path.split("/")[-1] in cls.detect_namelist:
                    index = cls.detect_namelist.index(image_path.split("/")[-1])
                    image_id = cls.detect_json_data["images"][index]["id"]

                # draw bounding boxes and labels of image detections
                if image_detection is not None:
                    # rescale boxes to original image
                    image_detection = rescale_boxes(image_detection, args.image_size, image.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in image_detection:
                        annotation_item_dict = {"image_id": image_id, "category_id": int(cls_pred) + 1,
                                                "bbox": [x1.item(), y1.item(), x2.item() - x1.item(),
                                                         y2.item() - y1.item()], "score": conf.item()}
                        annotation_list.append(annotation_item_dict)

            with timer("writing to {}".format(args.submission_file_path), logger):
                CommonUtils.get_json_file_by_object_func(target=annotation_list,
                                                         write_file_path=args.submission_file_path)

            return ResultUtils.success(data=CommonUtils.get_pickle_bytes_by_object_func({"detect": True}),
                                       media_type="application/octet-stream")


if __name__ == "__main__":
    Client.train(epoch=1)
