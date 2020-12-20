from fastapi import Response


class ResultUtils:
    """ unified request return class """

    @staticmethod
    def success(**k_args):
        res_data = {
            "code": 0,
            "msg": "request succeeded!"
        }

        if "data" in k_args and "media_type" not in k_args:
            res_data["data"] = k_args["data"]

        if "data" in k_args and "media_type" in k_args:
            return Response(content=k_args["data"], media_type=k_args["media_type"])

        if "data" not in k_args:
            res_data["data"] = None

        return res_data

    @staticmethod
    def error(**k_args):
        res_data = {
            "code": -1,
            "data": None
        }

        if "msg" in k_args:
            res_data["msg"] = k_args["msg"]
        else:
            res_data["msg"] = "request failed!"

        return res_data
