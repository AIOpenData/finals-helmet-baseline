import argparse


def args_parser():
    """ argument parser """
    parser = argparse.ArgumentParser()

    # federated parameters
    parser.add_argument("--server_epoch", type=int, default=20, help="number of server epochs: SE")
    parser.add_argument("--client_epoch", type=int, default=5, help="number of client epochs: CE")
    parser.add_argument("--user_num", type=int, default=2, help="number of users: K")

    # model parameters
    parser.add_argument("--model", type=str, default="yolov3", help="model name: M")
    parser.add_argument("--model_def", type=str,
                        default="config/finals_contest_helmet_federal_conf/yolov3_finals_contest_helmet_federal_conf.cfg",
                        help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default="/your/data/path/weights/darknet53.conv.74",
                        help="if specified starts from checkpoint model")
    parser.add_argument("--async_request_timeout", type=int, default=120, help="async request timeout: RT")
    parser.add_argument("--async_response_data_type", type=str, default="JSON",
                        help="async response data type ('JSON', 'READ' and 'TEXT'): DT")

    # other parameters
    parser.add_argument("--dataset", type=str, default="finals_contest_helmet_federal", help="name of dataset: D")
    parser.add_argument("--image_size", type=int, default=832, help="width and height of images")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="internet protocol of server")
    parser.add_argument("--server_port", type=int, default=5000, help="port of server")
    parser.add_argument("--client_ips", type=str, default="127.0.0.1,127.0.0.1",
                        help="internet protocols of all the clients")
    parser.add_argument("--client_ports", type=str, default="5001,5002", help="ports of all the clients")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    args = parser.parse_args()

    return args
