#!/usr/bin/env python
""" 0_segment_audio.vanilla.py -- Using a rolling window to split audio files into segments with a specific duration
Usage:
    0_segment_audio.vanilla.py [-hv] (-i <directory>) (-o <directory>) (-d <duration>) (-p <overlap>)
        [-a -l <labels.csv>] (-r <address>) (-s <password>) (-c <cores_per_node>) [-b <batch_size>]
        (-n <nodes>)

Positional Arguments:

Options:
    -h --help                           Print this screen and exit
    -v --version                        Print the version of 0_segment_audio.vanilla.py
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
    -n --num_nodes <nodes>              The number of nodes
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
from pathlib import Path
from splitter import Splitter
import torch
from os import environ
from timeit import default_timer as timer


@ray.remote(num_cpus=1)
def run_splitter(batch):
    dataset = Splitter(
        batch,
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
    for data in dataloader:
        for out in data:
            outputs.append(out)
    end = timer()

    print("DEBUG: end - start", end - start)

    return outputs


args = docopt(__doc__, version="0_segment_audio.vanilla.py version 0.0.1")

args["--duration"] = check_is_integer(args["--duration"], "--duration")
args["--overlap"] = check_is_integer(args["--overlap"], "--overlap")
args["--num_nodes"] = check_is_integer(
    args["--num_nodes"], "--num_nodes"
)
args["--cores_per_node"] = check_is_integer(
    args["--cores_per_node"], "--cores_per_node"
)

if args["--batch_size"]:
    args["--batch_size"] = check_is_integer(args["--batch_size"], "--batch_size")
else:
    args["--batch_size"] = 1

ray.init(address=args["--ray_address"], redis_password=args["--ray_password"])

input_p = Path(args["--input_directory"])
output_p = Path(args["--output_directory"])

all_wavs = list(input_p.rglob("**/*.WAV"))

batches = np.array_split(all_wavs, args["--num_nodes"])

ids = [run_splitter.remote(batch) for batch in batches]
results = ray.get(ids)
print(results)
