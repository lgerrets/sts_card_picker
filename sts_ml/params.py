
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
    },
}

params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2)