
card_predictor_params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1168512.data",
        "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1957.data",
        "batch_size": 256,
    },
    "model": {
        "dim": 256,
        "nheads": 4,
        # "ffdim": 512,
        "blocks": 4,
        "block_class": "MHALayer", # MHALayer, Linear
        "mask_inter_choices": False,
        "input_relics": True
    },
}

card_predictor_params["model"]["ffdim"] = card_predictor_params["model"].get("ffdim", card_predictor_params["model"]["dim"] * 2) # default to 2*dim if missing
card_predictor_params["model"]["input_relics"] = card_predictor_params["model"].get("input_relics", False) # default to False if missing

win_predictor_params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        # "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1168512.data",
        "dataset": "./SlayTheData_win_a10+_ic_64298_fb5dcd7_1957.data",
        "batch_size": 256,
    },
    "model": {
        "dim": 256,
        "nheads": 4,
        # "ffdim": 512,
        "blocks": 4,
        "block_class": "MHALayer", # MHALayer, Linear
        "input_relics": True,
        "gamma": 0.95,
    },
}

win_predictor_params["model"]["ffdim"] = win_predictor_params["model"].get("ffdim", win_predictor_params["model"]["dim"] * 2) # default to 2*dim if missing
win_predictor_params["model"]["input_relics"] = win_predictor_params["model"].get("input_relics", False) # default to False if missing
