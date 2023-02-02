
params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        "dataset": "./SlayTheData_win_a20_ic_21400_837d844_21399.data",
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

params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2) # default to 2*dim if missing
params["model"]["input_relics"] = params["model"].get("input_relics", False) # default to False if missing
