
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
        "block_class": "MHALayer", # MHALayer, Linear
        "enable_pad_mask": True,
    },
}

params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2)