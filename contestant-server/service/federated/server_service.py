import os
import time

import torch
from config.options_conf import args_parser
from http_request.async_request import AsyncRequest
from loguru import logger
from module.models_module import Darknet
from utils.common_utils import CommonUtils
from utils.tool_utils import set_random_seed, timer, weights_init_normal

# arguments parsing
args = args_parser()


class Server(object):
    """ the server class is responsible for scheduling each client
    to participate in federated training, testing and detecting """
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
    args.device = torch.device("cpu")

    args.model_file_path = os.path.join(args.model_folder_path,
                                        "D_{}_M_{}_SE_{}_CE_{}.pkl".format(args.dataset, args.model,
                                                                           args.server_epoch, args.client_epoch))
    args.log_file_path = os.path.join(args.log_folder_path,
                                      "D_{}_M_{}_SE_{}_CE_{}.log".format(args.dataset, args.model,
                                                                         args.server_epoch, args.client_epoch))

    # initialize log output configuration
    logger.add(args.log_file_path)

    # initiate model and load pretrained model weights
    model = Darknet(config_path=args.model_def, image_size=args.image_size).to(args.device)
    model.apply(weights_init_normal)

    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            model.load_darknet_weights(args.pretrained_weights)

    ip_lst = [ip for ip in args.client_ips.split(",")]
    port_lst = [port for port in args.client_ports.split(",")]
    federated_train_size_urls = ["http://{}:{}/federated_train_size".format(ip, port) for (ip, port) in
                                 zip(ip_lst, port_lst)]
    federated_train_urls = ["http://{}:{}/federated_train".format(ip, port) for (ip, port) in zip(ip_lst, port_lst)]
    federated_test_urls = ["http://{}:{}/federated_test".format(ip, port) for (ip, port) in zip(ip_lst, port_lst)]
    federated_detect_urls = ["http://{}:{}/federated_detect".format(ip, port) for (ip, port) in zip(ip_lst, port_lst)]

    server_model_params = model.state_dict()
    best_model_params = None
    client_ratio_lst = []

    @classmethod
    def call_async_request(cls, method: str = "", urls: list = [], data: dict = {}):
        """ async request for different methods
        (call_federated_train_size, call_federated_train, call_federated_test, call_federated_detect) """
        async_request = AsyncRequest()

        if method == "call_federated_train_size":
            url_info_list = [
                {
                    "url": url,
                    "method": "GET",
                    "task_name": url,
                    "response_data_type": "JSON",
                    "set_time_out": "false"
                }
                for url in urls
            ]
        else:
            url_info_list = [
                {
                    "url": url,
                    "method": "POST",
                    "data": data,
                    "task_name": url,
                    "response_data_type": "READ",
                    "set_time_out": "false"
                }
                for url in urls
            ]

        async_request.add_tasks_func(url_info_list)

        if method == "call_federated_train":
            async_request_cnt = 0
            while True:
                time.sleep(1)
                if len(async_request.task_info) != len(urls):
                    async_request_cnt += 1
                    if async_request_cnt % 600 == 0:
                        logger.info("async request for {} times, {} minutes, not finished!".format(async_request_cnt, async_request_cnt // 60))
                else:
                    break
        else:
            while True:
                time.sleep(1e-5)
                if len(async_request.task_info) == len(urls):
                    break

        async_request_result_dict = {}
        for url in urls:
            if method == "call_federated_train_size":
                async_request_result_dict[url] = async_request.task_info[url]["data"]["data"]
            else:
                async_request_result_dict[url] = CommonUtils.get_object_by_pickle_bytes_func(
                    async_request.task_info[url]["data"])
            async_request.destroy_task_func(url)

        return async_request_result_dict

    @classmethod
    def call_federated_train_size(cls):
        """ get the training data ratio of each client """
        with timer("call federated train size", logger):
            async_federated_train_size_request_result_dict = cls.call_async_request(method="call_federated_train_size",
                                                                                    urls=cls.federated_train_size_urls)
            for federated_train_size_url in cls.federated_train_size_urls:
                federated_train_size = async_federated_train_size_request_result_dict[federated_train_size_url][
                    "federated_train_size"]
                cls.client_ratio_lst.append(federated_train_size)

            logger.info("before normalization: client_ratio_lst: {}".format(cls.client_ratio_lst))
            client_ratio_sum = sum(cls.client_ratio_lst)
            cls.client_ratio_lst = [ratio / client_ratio_sum for ratio in cls.client_ratio_lst]
            logger.info("after normalization: client_ratio_lst: {}".format(cls.client_ratio_lst))

    @classmethod
    def call_federated_train(cls):
        """ call the model of each client for federated training """
        with timer("call federated train", logger):
            train_loss = []
            best_epoch = None
            best_loss = float("inf")

            for epoch in range(1, args.server_epoch + 1):
                with timer("train for epoch {}/{}".format(epoch, args.server_epoch), logger):
                    federated_train_param_dict = {"server_epoch": CommonUtils.get_pickle_bytes_by_object_func(epoch),
                                                  "server_model_params": CommonUtils.get_pickle_bytes_by_object_func(
                                                      cls.server_model_params)}
                    async_federated_train_request_result_dict = cls.call_async_request(method="call_federated_train",
                                                                                       urls=cls.federated_train_urls,
                                                                                       data=federated_train_param_dict)

                    avg_loss = 0.0
                    client_weight_lst = []

                    for idx, federated_train_url in enumerate(cls.federated_train_urls):
                        returned_client_model_params = async_federated_train_request_result_dict[federated_train_url][
                            "client_model_params"]
                        returned_epo_avg_loss = async_federated_train_request_result_dict[federated_train_url][
                            "epo_avg_loss"]

                        # update the average training loss of all clients for the epoch
                        avg_loss += (returned_epo_avg_loss - avg_loss) / (idx + 1)

                        client_weight_lst.append(returned_client_model_params)

                    for key in client_weight_lst[-1].keys():
                        client_weight_lst[-1][key] = cls.client_ratio_lst[-1] * client_weight_lst[-1][key]
                        for idx in range(0, len(client_weight_lst) - 1):
                            client_weight_lst[-1][key] += cls.client_ratio_lst[idx] * client_weight_lst[idx][key]

                    cls.server_model_params = client_weight_lst[-1]

                    logger.info("epoch {:3d}, average loss {:.3f}".format(epoch, avg_loss))
                    train_loss.append(avg_loss)

                    # save the model, loss and epoch with the smallest training average loss for all the epochs
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_epoch = epoch
                        cls.best_model_params = cls.server_model_params

            logger.info("best train loss: {}".format(best_loss))
            logger.info("best epoch: {}".format(best_epoch))
            CommonUtils.get_pickle_file_by_object_func(target=cls.best_model_params,
                                                       write_file_path=args.model_file_path)

    @classmethod
    def call_federated_test(cls):
        """ send the best model to all the clients for testing after the federated training """
        with timer("call federated test", logger):
            federated_test_param_dict = {
                "best_model_params": CommonUtils.get_pickle_bytes_by_object_func(cls.best_model_params)}
            cls.call_async_request(method="call_federated_test", urls=cls.federated_test_urls,
                                   data=federated_test_param_dict)

    @classmethod
    def call_federated_detect(cls):
        """ send the best model to all the clients for detecting after the federated training """
        with timer("call federated detect", logger):
            federated_detect_param_dict = {
                "best_model_params": CommonUtils.get_pickle_bytes_by_object_func(cls.best_model_params)}
            cls.call_async_request(method="call_federated_detect", urls=cls.federated_detect_urls,
                                   data=federated_detect_param_dict)


if __name__ == "__main__":
    Server.call_federated_train_size()
    Server.call_federated_train()
    Server.call_federated_test()
    Server.call_federated_detect()
