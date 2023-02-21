from typing import Union
import os
import json

from sts_ml.model_abc import PARAMS_FILENAME, ModelAbc
from sts_ml.model import CardModel
from sts_ml.win_model import WinModel

def instanciate_model(params: dict = None, training_dir: str = None, ckpt: int = None) -> Union[CardModel, WinModel]:
    do_load_ckpt = training_dir is not None
    do_create_model = params is not None
    assert (training_dir is None) >= (ckpt is None)
    assert do_load_ckpt != do_create_model

    if params is None:
        params = json.load(open(os.path.join(training_dir, PARAMS_FILENAME), "r"))

    model_cls_str = params["model"]["model_cls"]

    if model_cls_str == "CardModel":
        model_cls = CardModel
    elif model_cls_str == "WinModel":
        model_cls = WinModel
    else:
        assert False

    if do_load_ckpt:
        model = model_cls.load_model(training_dir, ckpt)
    elif do_create_model:
        model, train_dataloader, val_dataloader = model_cls.create_model(params)
    else:
        assert False
    
    return model

