
params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
        "dataset": "./SlayTheData_win_a20_ic_21400.data",
    },
    "model": {
        "dim": 256,
        # "ffdim": 512,
        "blocks": 4,
        "read_relics": False,
        "block_class": "MHALayer", # MHALayer, Linear
    },
}

params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2)