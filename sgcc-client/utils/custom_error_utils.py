from loguru import logger


class CustomError(Exception):
    """ custom error class """
    def __init__(self, msg):
        super().__init__(self)
        self.msg = msg

    def __str__(self):
        return self.msg


if __name__ == "__main__":
    try:
        raise CustomError("custom error!")
    except CustomError as e:
        logger.error(e)
