
params = {
    "train": {
        "lr": 1e-4,
        "split": 0.8,
    },
    "model": {
        "dim": 256,
        # "ffdim": 512,
        "blocks": 4,
    },
}

params["model"]["ffdim"] = params["model"].get("ffdim", params["model"]["dim"] * 2)