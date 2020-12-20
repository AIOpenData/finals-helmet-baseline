import uvicorn
import config.base_conf as base_conf
import config.routes_conf as routes_conf

from config.options_conf import args_parser
from fastapi import FastAPI
from loguru import logger

app = FastAPI()
args = args_parser()

for i in routes_conf.routes:
    app.include_router(router=i.router)


class CustomFastApi:
    def __init__(self):
        logger.info("initialize project")
        self.host = base_conf.env["host"]
        self.port = base_conf.env["port"]
        logger.info("Host: {0}".format(self.host))
        logger.info("Port: {0}".format(self.port))

    def run(self):
        logger.info("start project")
        uvicorn.run(app="main:app", host=self.host, port=self.port, reload=False)


if __name__ == "__main__":
    CustomFastApi().run()
