from fastapi import APIRouter, File
from loguru import logger
from service.federated.client_service import Client
from utils.common_utils import CommonUtils
from utils.result_utils import ResultUtils

router = APIRouter()


@router.get("/federated_train_size")
def federated_train_size():
    return Client.get_federated_train_size()


@router.post("/federated_train")
def federated_train(server_epoch: bytes = File(...), server_model_params: bytes = File(...)):
    # receive the server training epoch and initial or federated averaging model
    server_epoch = CommonUtils.get_object_by_pickle_bytes_func(server_epoch)
    server_model_params = CommonUtils.get_object_by_pickle_bytes_func(server_model_params)

    # return the local model after training of current client to server
    return Client.train(server_model_params=server_model_params, epoch=server_epoch)


@router.post("/federated_test")
def federated_test(best_model_params: bytes = File(...)):
    # receive the final best model from server and do the evaluating
    best_model_params = CommonUtils.get_object_by_pickle_bytes_func(best_model_params)

    return Client.test(test_model_params=best_model_params, mode="test")


@router.post("/federated_detect")
def federated_detect(best_model_params: bytes = File(...)):
    # receive the final best model from server and do the evaluating
    best_model_params = CommonUtils.get_object_by_pickle_bytes_func(best_model_params)

    return Client.detect(detect_model_params=best_model_params)
