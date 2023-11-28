# %%
from datetime import datetime
import json
import glob
import os
from pathlib import Path

import torch
from torch import nn

# %%
def write_event(log, mode, epoch: int, **data):
    data["mode"] = mode
    data["epoch"] = epoch
    data["dt"] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write("\n")
    log.flush()


def load_model(model: nn.Module, path: Path) -> dict:
    print("PATH_STR: ",str(path))
    state = torch.load(str(path),map_location=torch.device('cpu'))
    model.load_state_dict(state["model"])
    print("Loaded model from epoch {epoch}".format(**state))
    return state
