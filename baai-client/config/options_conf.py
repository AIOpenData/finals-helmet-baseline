# -*- coding: utf-8 -*-
import argparse


def args_parser():
    """ argument parser """
    parser = argparse.ArgumentParser()

    # federated parameters
    parser.add_argument('--server_epoch', type=int, default=20, help="number of server epochs: SE")
    parser.add_argument('--client_epoch', type=int, default=5, help="number of client epochs: CE")
    parser.add_argument('--user_num', type=int, default=2, help="number of users: K")
    parser.add_argument('--user_id', type=int, default=1, help="id of current user: ID")
    parser.add_argument('--train_batch_size', type=int, default=8, help="client training batch size: TB")
    parser.add_argument('--valid_batch_size', type=int, default=8, help="client valid batch size: VB")
    parser.add_argument('--test_batch_size', type=int, default=8, help="client test batch size: CB")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate: LR")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum: MM")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="weight decay: WD")
    parser.add_argument("--gradient_accumulations", type=int, default=2,
                        help="number of gradient accumulations before step: GA")

    # model parameters
    parser.add_argument('--model', type=str, default='yolov3', help='model name: M')
    parser.add_argument("--model_def", type=str,
                        default="config/finals_contest_helmet_federal_conf"
                                "/yolov3_finals_contest_helmet_federal_conf.cfg",
                        help="path to model definition file: MD")

    # other parameters
    parser.add_argument('--dataset', type=str, default='finals_contest_helmet_federal_conf',
                        help="name of dataset: D")
    parser.add_argument("--data_config", type=str,
                        default="config/finals_contest_helmet_federal_conf"
                                "/finals_contest_helmet_federal_conf.data",
                        help="path to data config file: DC")
    parser.add_argument('--train_prop', type=float, default=0.7, help='training dataset proportion: TP')
    parser.add_argument('--image_size', type=int, default=832, help='width and height of images: IS')
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges: NC")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold: IT")  # 0.001
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="object confidence threshold: CT")  # 0.001
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="iou threshold for non-maximum suppression: NT")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument('--server_ip', type=str, default="127.0.0.1", help="internet protocol of server: SI")
    parser.add_argument('--server_port', type=int, default=5000, help="port of server: SP")
    parser.add_argument('--client_ip', type=str, default="127.0.0.1", help="internet protocol of client: CI")
    parser.add_argument('--client_port', type=int, default=5001, help="port of client: CP")
    parser.add_argument('--verbose', type=bool, default=True, help='verbose print: VP')
    parser.add_argument('--log_interval', type=int, default=50,
                        help="the number of steps as an interval for logging: LI")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights: CI")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set: EI")
    parser.add_argument("--multi_scale_training", default=True, help="allow for multi-scale training: MST")

    args = parser.parse_args()

    return args
