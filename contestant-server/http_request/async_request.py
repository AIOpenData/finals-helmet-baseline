import time
import asyncio
import numpy as np

from aiohttp import ClientSession, ClientTimeout
from config.options_conf import args_parser
from loguru import logger
from threading import Thread
from utils.custom_error_utils import CustomError
from utils.common_utils import CommonUtils

args = args_parser()


def start_loop_func(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class AsyncRequest:
    def __init__(self):
        """ start event loop and thread """
        # logger.info("Start Async Request")
        self.event_loop = asyncio.new_event_loop()
        self.t = Thread(target=start_loop_func, args=(self.event_loop,))
        self.t.setDaemon(True)
        self.t.start()
        self.task_info = {}

    @staticmethod
    async def request_func(url, method, headers, data, task_name, response_data_type, timeout):
        """ send the request """
        logger.info("start to execute task name: {}".format(task_name))

        async with ClientSession(timeout=ClientTimeout(total=timeout)) as session:
            try:
                start_time = time.time()

                if method == "GET":
                    async with session.get(url=url, headers=headers) as response:
                        if response_data_type == "READ":
                            content = await response.read()
                        elif response_data_type == "JSON":
                            content = await response.json()
                        elif response_data_type == "TEXT":
                            content = await response.text()
                        else:
                            raise CustomError(
                                "task name: {0}, please set the standard asynchronous request response type".format(
                                    task_name))

                        return {
                            "code": response.status,
                            "data": content,
                            "task_name": task_name,
                            "response_data_type": response_data_type,
                            "request_time": format(time.time() - start_time, '.4f') + "s"
                        }
                elif method == "POST":
                    async with session.post(url=url, data=data, headers=headers) as response:
                        if response_data_type == "READ":
                            content = await response.read()
                        elif response_data_type == "JSON":
                            content = await response.json()
                        elif response_data_type == "TEXT":
                            content = await response.text()
                        else:
                            raise CustomError(
                                "task name: {0}, please set the standard asynchronous request response type".format(
                                    task_name))

                        return {
                            "code": response.status,
                            "data": content,
                            "task_name": task_name,
                            "response_data_type": response_data_type,
                            "request_time": format(time.time() - start_time, '.4f') + "s"
                        }
            except Exception as e:
                raise CustomError("task name: {0}, error in request: {1}".format(task_name, e))

    def callback_func(self, future):
        """ call back function """
        code = future.result()["code"]
        data = future.result()["data"]
        task_name = future.result()["task_name"]
        response_data_type = future.result()["response_data_type"]
        request_time = future.result()["request_time"]

        if future.exception():
            response = {
                "code": -1,
                "msg": "request failed! exception: {}".format(str(future.exception())),
                "task_name": task_name,
                "response_data_type": response_data_type,
                "request_time": request_time
            }
            raise CustomError("task name: {0}, msg: {1}".format(task_name, response["msg"]))
        else:
            if not code:
                response = {
                    "code": -1,
                    "msg": "unknown error!",
                    "task_name": task_name,
                    "response_data_type": response_data_type,
                    "request_time": request_time
                }
                raise CustomError("task name: {0}, msg: {1}".format(task_name, response["msg"]))
            else:
                if code != 200:
                    response = {
                        "code": -1,
                        "msg": "request failed! status code: {}".format(str(code)),
                        "task_name": task_name,
                        "response_data_type": response_data_type,
                        "request_time": request_time
                    }
                    raise CustomError("task name: {0}, msg: {1}".format(task_name, response["msg"]))
                else:
                    response = {
                        "code": 200,
                        "msg": "request succeeded!",
                        "data": data,
                        "task_name": task_name,
                        "response_data_type": response_data_type,
                        "request_time": request_time
                    }
                    self.task_info[response["task_name"]] = response
                    logger.info("task name: {0}, msg: {1}, response has been already stored in memory!".format(task_name, response["msg"]))

    def destroy_task_func(self, task_name):
        """ destroy the used task name to ensure that the memory will not crash """
        self.task_info.pop(task_name)

    def add_tasks_func(self, task_list):
        """ add tasks """
        for task_info_dict in task_list:
            checked_task_info_dict = self.check_task_info_func(task_info_dict)

            future = asyncio.run_coroutine_threadsafe(
                self.request_func(url=checked_task_info_dict["url"], method=checked_task_info_dict["method"],
                                  headers=checked_task_info_dict["headers"],
                                  data=checked_task_info_dict["data"], task_name=checked_task_info_dict["task_name"],
                                  response_data_type=checked_task_info_dict["response_data_type"],
                                  timeout=checked_task_info_dict["set_time_out"]),
                self.event_loop)

            # add the call back function for the future object
            future.add_done_callback(self.callback_func)

    @staticmethod
    def check_task_info_func(task_info: dict):
        if "url" in task_info and task_info["url"]:
            url = task_info["url"]
        else:
            raise CustomError("url does not existed!")

        if "method" in task_info and task_info["method"]:
            method = task_info["method"]
        else:
            method = "GET"

        if "headers" in task_info and task_info["headers"]:
            headers = task_info["headers"]
        else:
            headers = None

        if "data" in task_info and task_info["data"]:
            if type(task_info["data"]) != dict:
                raise CustomError("data type should be dict!")
            else:
                data = task_info["data"]
                data_list = list(enumerate(data))

                if method == "GET":
                    for i in data_list:
                        if i[0] == 0:
                            url += "?"

                        if i[0] != len(data_list) - 1:
                            url += i[1] + "=" + data[i[1]] + "&"
                        else:
                            url += i[1] + "=" + data[i[1]]
        else:
            data = None

        if "task_name" in task_info and task_info["task_name"]:
            task_name = task_info["task_name"]
        else:
            raise CustomError("{0} does not existed!".format("task_name"))

        if "response_data_type" in task_info and task_info["response_data_type"]:
            response_data_type = task_info["response_data_type"]
        else:
            response_data_type = args.async_response_data_type

        if "set_time_out" in task_info and task_info["set_time_out"]:
            if type(task_info["set_time_out"]) == str:
                if task_info["set_time_out"] == "false":
                    set_time_out = 0
                elif task_info["set_time_out"] == "true":
                    set_time_out = args.async_request_timeout
                else:
                    raise CustomError("{0} can only be: 'true' or 'false'!".format("set_time_out"))
            else:
                raise CustomError("{0} data type error!".format("set_time_out"))
        else:
            set_time_out = args.async_request_timeout

        return {
            "url": url,
            "method": method,
            "headers": headers,
            "data": data,
            "task_name": task_name,
            "response_data_type": response_data_type,
            "set_time_out": set_time_out
        }


if __name__ == "__main__":
    logger.info("start to execute!")
    async_request = AsyncRequest()
    target_url_info_list = [
        {
            "url": "http://127.0.0.1:5001/example1",
            "method": "GET",
            "data": {
                "str_param": "Tom"
            },
            "task_name": "http://127.0.0.1:5001/example1",
            "response_data_type": "JSON",
            "set_time_out": "true"
        },
        {
            "url": "http://127.0.0.1:5001/example2",
            "method": "POST",
            "data": {
                "numpy_bytes_param": CommonUtils.get_pickle_bytes_by_object_func(np.array([1, 2, 3, 4]))
            },
            "task_name": "http://127.0.0.1:5001/example2",
            "response_data_type": "READ",
            "set_time_out": "false"
        }
    ]

    async_request.add_tasks_func(target_url_info_list)

    url_list = ['http://127.0.0.1:5001/example1', 'http://127.0.0.1:5001/example2']

    while True:
        empty_cnt = 0
        for url in url_list:
            if url not in async_request.task_info:
                empty_cnt += 1
                print("{} has not been finished!".format(url))
        if empty_cnt == 0:
            break
        time.sleep(1)

    print("async_request task_info: {}\n".format(async_request.task_info))
