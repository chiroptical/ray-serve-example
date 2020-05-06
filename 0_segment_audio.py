#!/usr/bin/env python
""" 0-segment-audio.ray.py -- Using a rolling window to split audio files into segments with a specific duration
Usage:
    0-segment-audio.ray.py [-hv] (-i <directory>) (-o <directory>) (-d <duration>) (-p <overlap>)
        [-a -l <labels.csv>] (-r <address>) (-s <password>) (-c <cores_per_node>) [-b <batch_size>]

Positional Arguments:

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version of 0-segment-audio.ray.py
    -i --input_directory <directory>    The input directory to search for audio files (`.{wav,WAV}`)
    -o --output_directory <directory>   The output directory for segments
    -d --duration <duration>            The segment duration in seconds
    -p --overlap <overlap>              Overlap of each segment in seconds
    -a --annotations                    Only include segments which overlap with Raven annotations
                                        -> Audio files without annotations are skipped when using this argument
    -l --labels <labels.csv>            When using `--annotations`, the `labels.csv` file provides a map from an
                                            incorrect label (column `from`) to the corrected label (column `to`)
    -r --ray_address <address>          The ray cluster address
    -s --ray_password <password>        The ray cluster password
    -c --cores_per_node <cores>         Number of cores per node
    -b --batch_size <batch_size>        The batch size [default: 1]
"""


def check_is_integer(x, parameter):
    try:
        r = int(x)
        if r <= 0:
            raise ValueError
        return r
    except ValueError:
        exit(f"Error: `{parameter}` should be a positive whole number! Got `{x}`")


from docopt import docopt
import numpy as np
import ray
from ray import serve
from pathlib import Path
from splitter import Splitter
import torch
from os import environ
from timeit import default_timer as timer


class RunSplitter:
    def __init__(self):
        pass

    @serve.accept_batch
    def __call__(self, _, *, audio_paths=[]):
        batch_size = len(audio_paths)
        if batch_size == 0:
            return []

        dataset = Splitter(
            audio_paths,
            annotations=args["--annotations"],
            labels=args["--labels"],
            overlap=args["--overlap"],
            duration=args["--duration"],
            output_directory=args["--output_directory"],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            # batch_size=batch_size,
            batch_size=1,
            shuffle=False,
            num_workers=args["--cores_per_node"],
            collate_fn=dataset.collate_fn,
        )

        start = timer()
        outputs = []
        for idx, data in enumerate(dataloader):
            for out in data:
                outputs.append(out)
        end = timer()
        print("DEBUG: end - start", end - start)

        return outputs


args = docopt(__doc__, version="0_segment_audio.ray.py version 0.0.1")

args["--duration"] = check_is_integer(args["--duration"], "--duration")
args["--overlap"] = check_is_integer(args["--overlap"], "--overlap")
args["--cores_per_node"] = check_is_integer(
    args["--cores_per_node"], "--cores_per_node"
)

if args["--batch_size"]:
    args["--batch_size"] = check_is_integer(args["--batch_size"], "--batch_size")
else:
    args["--batch_size"] = 1

ray.init(address=args["--ray_address"], redis_password=args["--ray_password"])
serve.init(start_server=False)

input_p = Path(args["--input_directory"])
output_p = Path(args["--output_directory"])

all_wavs = list(input_p.rglob("**/*.WAV"))

# model = RunSplitter()
# predictions = model(None, audio_paths=all_wavs[0:10])
# print(predictions)

serve.create_endpoint("splitter")
serve.create_backend(
    RunSplitter,
    "splitter:v0",
    backend_config=serve.BackendConfig(
        num_replicas=2, max_batch_size=args["--batch_size"]
    ),
)
serve.link("splitter", "splitter:v0")

handle = serve.get_handle("splitter")

ids = [handle.remote(audio_paths=audio_path) for audio_path in all_wavs]
results = ray.get(ids)
print(results)
