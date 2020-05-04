import torch
import pandas as pd
from math import ceil, floor
from hashlib import md5
from librosa.output import write_wav
from librosa.core import load, get_duration
import sys
from pathlib import Path
import numpy as np
from itertools import chain


def get_segment(clip_begin, clip_end, samples, sr):
    begin = floor(clip_begin * sr)
    end = ceil(clip_end * sr)

    return samples[begin:end], begin, end


def get_md5_digest(s):
    obj = md5()
    obj.update(s.encode("utf-8"))
    return obj.hexdigest()


def annotations_with_overlaps_with_clip(df, begin, end):
    return df[
        ((df["begin time (s)"] >= begin) & (df["begin time (s)"] < end))
        | ((df["end time (s)"] > begin) & (df["end time (s)"] <= end))
    ]


class Splitter(torch.utils.data.Dataset):
    def __init__(
        self,
        wavs,
        annotations=None,
        labels=None,
        overlap=1,
        duration=5,
        output_directory="segments",
    ):
        self.wavs = list(wavs)

        self.annotations = annotations
        self.labels = labels
        if self.labels:
            self.labels_df = pd.read_csv(labels)

        self.overlap = overlap
        self.duration = duration
        self.output_directory = output_directory

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, item_idx):
        wav = self.wavs[item_idx]
        annotation_prefix = self.wavs[item_idx].stem.split(".")[0]

        if self.annotations:
            annotation_file = Path(
                f"{wav.parent}/{annotation_prefix}.Table.1.selections.txt.lower"
            )
            if not annotation_file.is_file():
                sys.stderr.write(f"Warning: Found no Raven annotations for {wav}\n")
                return {"data": []}

        # TODO: Need to feed audio related configurations to `load`
        wav_samples, wav_sample_rate = load(wav)
        wav_duration = get_duration(wav_samples, sr=wav_sample_rate)
        wav_times = np.arange(0.0, wav_duration, wav_duration / len(wav_samples))

        if self.annotations:
            annotation_df = pd.read_csv(annotation_file, sep="\t").sort_values(
                by=["begin time (s)"]
            )

        if self.labels:
            annotation_df["class"] = annotation_df["class"].fillna("unknown")
            annotation_df["class"] = annotation_df["class"].apply(
                lambda cls: self.labels_df[self.labels_df["from"] == cls]["to"].values[
                    0
                ]
            )

        num_segments = ceil(
            (wav_duration - self.overlap) / (self.duration - self.overlap)
        )

        outputs = []
        for idx in range(num_segments):
            if idx == num_segments - 1:
                end = wav_duration
                begin = end - self.duration
            else:
                begin = self.duration * idx - self.overlap * idx
                end = begin + self.duration

            if self.annotations:
                overlaps = annotations_with_overlaps_with_clip(
                    annotation_df, begin, end
                )

            unique_string = f"{wav}-{begin}-{end}"
            destination = f"{self.output_directory}/{get_md5_digest(unique_string)}"

            if self.annotations:
                if overlaps.shape[0] > 0:
                    segment_samples, segment_sample_begin, segment_sample_end = get_segment(
                        begin, end, wav_samples, wav_sample_rate
                    )
                    write_wav(f"{destination}.WAV", segment_samples, wav_sample_rate)

                    if idx == num_segments - 1:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.WAV"
                    else:
                        to_append = f"{wav},{annotation_file},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.WAV"
                    to_append += f",{'|'.join(overlaps['class'].unique())}"

                    outputs.append(to_append)
            else:
                segment_samples, segment_sample_begin, segment_sample_end = get_segment(
                    begin, end, wav_samples, wav_sample_rate
                )
                write_wav(f"{destination}.WAV", segment_samples, wav_sample_rate)

                if idx == num_segments - 1:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[-1]},{destination}.WAV"
                else:
                    to_append = f"{wav},{wav_times[segment_sample_begin]},{wav_times[segment_sample_end]},{destination}.WAV"

                outputs.append(to_append)

        return {"data": outputs}

    def collate_fn(*batch):
        return chain.from_iterable([x["data"] for x in batch[1]])
