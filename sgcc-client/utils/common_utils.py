import json
import pickle

from loguru import logger


class CommonUtils:
    """ common tools """

    @staticmethod
    def print_msg_func(msg):
        logger.info(" ======> {} ".format(msg))

    @staticmethod
    def merge_dict_func(dict_1, dict_2):
        """ merge two dicts into one dict """
        return dict_2.update(dict_1)

    @staticmethod
    def get_json_str_by_object_func(target: object):
        """ get the json formatted str by the object
        (set ensure_ascii=False for preventing Chinese error codes) """
        return json.dumps(obj=target, ensure_ascii=False, indent=5)

    @staticmethod
    def get_object_by_json_str_func(string: str):
        """ get the object by the json formatted str """
        return json.loads(s=string)

    @staticmethod
    def get_json_file_by_object_func(target: object, write_file_path: str):
        """ get the json file by the object """
        return json.dump(obj=target, fp=open(write_file_path, mode="w", encoding="utf-8"), ensure_ascii=False, indent=5)

    @staticmethod
    def get_object_by_json_file_func(read_file_path: str):
        """ get the object by the json file """
        return json.load(fp=open(read_file_path, mode="r", encoding="utf-8"))

    @staticmethod
    def get_pickle_bytes_by_object_func(target: object):
        """ get the pickled bytes representation by the object """
        return pickle.dumps(obj=target)

    @staticmethod
    def get_object_by_pickle_bytes_func(byte_str: bytes):
        """ get the object by the pickled bytes representation """
        return pickle.loads(data=byte_str)

    @staticmethod
    def get_pickle_file_by_object_func(target: object, write_file_path: str):
        """ get the pickled bytes file by the object """
        return pickle.dump(obj=target, file=open(write_file_path, mode="wb"))

    @staticmethod
    def get_object_by_pickle_file_func(read_file_path: str):
        """ get the object by the pickled bytes file """
        return pickle.load(file=open(read_file_path, mode="rb"))
