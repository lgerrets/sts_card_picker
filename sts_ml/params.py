import os

NUM_DATALOADER_WORKERS = 0 # set this to 0 or >= 2

card_predictor_params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1168512.data",
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1957.data",
        "dataset": [
            "./SlayTheData_a10+_ic_2_128944_944114b_3712932",
        ],
        "batch_size": 256,
        "data_sampling": "round", # one of "uniform", "round"
    },
    "model": {
        "model_cls": "CardModel",
        "dim": 256,
        "nheads": 4,
        # "ffdim": 512,
        "blocks": 4,
        "block_class": "MHALayer", # MHALayer, Linear
        "mask_inter_choices": False,
        "input_relics": True
    },
}

win_predictor_params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1168512.data",
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1957.data",
        "dataset": [
            "./SlayTheData_a10+_ic_2_128944_944114b_3712932",
        ],
        "batch_size": 256,
        "data_sampling": "round", # one of "uniform", "round"
    },
    "model": {
        "model_cls": "WinModel",
        "dim": 256,
        "nheads": 4,
        # "ffdim": 512,
        "blocks": 4,
        "block_class": "MHALayer", # MHALayer, Linear
        "input_relics": True,
        "gamma": 0.95,
    },
}

for params in [card_predictor_params, win_predictor_params]:
    params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2) # default to 2*dim if missing
    params["model"]["input_relics"] = params["model"].get("input_relics", False) # default to False if missing
    if os.path.isdir(params["train"]["dataset"][0]):
        assert len(params["train"]["dataset"]) == 1
        dirpath = params["train"]["dataset"][0]
        filenames = os.listdir(dirpath)
        params["train"]["dataset"] = [os.path.join(dirpath, filename) for filename in filenames]
