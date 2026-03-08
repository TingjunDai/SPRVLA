import json
import logging
import os
import re
import shutil
from os.path import join, exists
from typing import Iterable
from collections import defaultdict
import random

import datasets
import numpy as np
import torchvision
from cached_path import cached_path
from torchvision.transforms import functional as VF
from PIL import ImageOps
from torchvision.transforms.functional import affine, InterpolationMode

from olmo.data.dataset import DATA_HOME, Dataset, DatasetBase
from olmo.data.download_urls import download_pixmo_urls, filter_and_group_data, add_internal_urls
from olmo.data.image_preprocessor import load_pil_image, save_images
from olmo.util import transpose_dict_of_lists

if DATA_HOME is not None:
    PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
    COSYN_IMAGES = join(DATA_HOME, "cosyn_images")
else:
    PIXMO_DATASETS = None
    COSYN_IMAGES = None
"""Where to save local version of the data after URLs filtering"""


VERIFY = True
"""Verify SSL certificates when downloading"""


NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
"""No-pointing requests templates, used for preprocessing"""


def save_local_dataset(dataset: datasets.Dataset, name: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info("Preparing local dataset...")
    if exists(name):
        logging.info(f"{name} already exists, it will be removed")
        shutil.rmtree(name)
    dataset.save_to_disk(name, num_proc=n_procs)
    logging.info("Done")


class PixMoCount(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "count")
        if exists(local_name):
            return
        all_data = datasets.DatasetDict()
        for split in ["validation", "test", "train"]:
            ds = datasets.load_dataset("allenai/pixmo-count", split=split)
            url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=False)
            ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
            ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, sample=None, counting=False, keep_in_memory=False):
        self.dataset = datasets.load_from_disk(join(PIXMO_DATASETS, "count"), keep_in_memory=keep_in_memory)[split]
        self.counting = counting
        self.split = split

    def __len__(self):
        if self.counting == "both":
            return len(self.dataset) * 2
        else:
            return len(self.dataset)

    def get(self, item, rng):
        if self.counting == "both":
            mode = "point_count" if (item%2==0) else "pointing"
            item = item // 2
        else:
            mode = "point_count" if self.counting else "pointing"

        example = self.dataset[item]
        out = dict(
            style=mode,
            image=example["image"],
            label=example["label"],
            metadata=dict(
                image_url=example["image_url"],
                count=example["count"],
            )
        )
        if self.split == "train":
            points = example["points"]
            out["points"] = np.stack([points["x"], points["y"]], -1, dtype=np.float32)
        return out


class PixMoDocs(Dataset):

    @staticmethod
    def save_image(images: Iterable):
        raise NotImplementedError()
        keys = []
        for image in images:
            key = compute_hash(image["bytes"])
            keys.append(key)
            with open(join(DATA_HOME, "pixmo_docs_images", key), "wb") as f:
                f.write(image["bytes"])
        return dict(image_path=keys)

    @classmethod
    def download(cls, n_procs=1):
        for name in ["other", "charts", "diagrams", "tables"]:
            local_name = join(PIXMO_DATASETS, f"pixmo_docs_{name}")
            if exists(local_name):
                continue
            datasets.load_dataset_builder("allenai/pixmo-docs", name=name).download_and_prepare()
            all_data = datasets.DatasetDict()
            for split in ["validation", "train"]:
                ds = datasets.load_dataset("allenai/pixmo-docs", split=split, name=name)
                ds = ds.cast_column("image", datasets.Image(decode=False))
                # Doing this inplace causes issue with the column feature type,
                # so just map to a new column and then replace the old one
                ds = ds.map(
                    cls.save_image,
                    input_columns="image",
                    batched=True,
                    batch_size=256,
                    num_proc=n_procs if len(ds) > 10000 else 1,
                    desc=f"{name}-{split}-images",
                    remove_columns="image",
                    load_from_cache_file=False
                )
                ds = ds.rename_column("image_path", "image")
                all_data[split] = ds
            save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, doc_type, split, sample=None, keep_in_memory=False, flat=False, use_image_files=True):
        assert doc_type in ["other", "charts", "diagrams", "tables"]
        assert split in ["train", "validation"]
        self.doc_type = doc_type
        self.flat = flat
        self.use_image_files = use_image_files
        if use_image_files:
            # Load a local version of the data that contains filenames instead of the images directly
            local_name = join(PIXMO_DATASETS, f"pixmo_docs_{doc_type}")
            self.dataset = datasets.load_from_disk(local_name, keep_in_memory=keep_in_memory)[split]
        else:
            self.dataset = datasets.load_dataset(
                "allenai/pixmo-docs", name=doc_type, split=split, keep_in_memory=keep_in_memory)
        if flat:
            # Use an index so we don't have to load the images into memory if `keep_in_memory=False`
            # FIXME just switch to the JSON dataset
            logging.info("Building flat index")
            offset = 0
            n_questions = [len(x["question"]) for x in self.dataset["questions"]]
            image_index = np.repeat(np.arange(len(self.dataset), dtype=np.int32), n_questions)
            question_index = np.concatenate([np.arange(x, dtype=np.int32) for x in n_questions], 0)
            self.flat_index = np.stack([image_index, question_index], 1)
            logging.info("Done")

    def __len__(self):
        return len(self.flat_index) if self.flat else len(self.dataset)

    def get(self, item, rng):
        style = f"pixmo_docs_{self.doc_type}"
        if self.flat:
            image_ix, question_ix = self.flat_index[item]
            example = self.dataset[int(image_ix)]
            if self.use_image_files:
                example["image"] = join(DATA_HOME, "pixmo_docs_images", example["image"])
            qas = example["questions"]
            return dict(
                image=example["image"],
                question=qas["question"][question_ix],
                answer=qas["answer"][question_ix],
                style=style,
                metadata=dict(
                    image_id=example["image_id"]
                )
            )
        example = self.dataset[item]
        qas = example["questions"]
        if self.use_image_files:
            example["image"] = join(DATA_HOME, "pixmo_docs_images", example["image"])
        return dict(
            image=example["image"],
            message_list=[
                dict(question=q, answer=a, style=style) for q, a in
                zip(qas["question"], qas["answer"])
            ],
            metadata=dict(
                image_id=example["image_id"]
            )
        )


class PixMoPoints(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False, hold_out_pointing_eval=True):
        collection_method = ["pointing", "counting"]
        local_names = [join(PIXMO_DATASETS, f"points-{name}") for name in collection_method]
        if all(exists(x) for x in local_names):
            return
        ds = datasets.load_dataset("allenai/pixmo-points", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        if hold_out_pointing_eval:
            eval_ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
            for url in eval_ds["image_url"]:
                if url in filenames:
                    del filenames[url]
        for method, local_name in zip(collection_method, local_names):
            logging.info(f"Building subset {method}")
            ds_for_method = ds.filter(lambda x: x == method, input_columns="collection_method")
            filtered_dataset = filter_and_group_data(ds_for_method, filenames, check_sha)
            name = "high_frequency" if method == "counting" else "basic"
            save_local_dataset(filtered_dataset, local_name, n_procs=n_procs, n_val=n_val)

    def __init__(self, split, kind="both", counting=False, keep_in_memory=False,
                 max_points=None, max_total_points_per_example=None, max_samples_per_image=None):
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.counting = counting
        if counting == "both":
            self.mode = ["point_count", "pointing"]
        else:
            self.mode = "point_count" if counting else "pointing"
        self.split = split
        self.kind = kind
        if kind == "both":
            data1 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-counting"), keep_in_memory=keep_in_memory)[split]
            data2 = datasets.load_from_disk(
                join(PIXMO_DATASETS, "points-pointing"), keep_in_memory=keep_in_memory)[split]
            self.data = datasets.concatenate_datasets([data1, data2])
        elif kind == "basic":
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-pointing"), keep_in_memory=keep_in_memory)[split]
        else:
            self.data = datasets.load_from_disk(
                join(PIXMO_DATASETS, f"points-counting"), keep_in_memory=keep_in_memory)[split]
        if max_total_points_per_example or max_points:
            data = transpose_dict_of_lists(self.data[:])
            flattened = []
            n_filtered = 0
            total_points = 0
            for ex in data:
                sub_batches = []
                on = []
                total_on = 0
                total_points += len(ex["points"])
                for ix, points in enumerate(ex["points"]):
                    n = len(points)
                    if max_points and n > max_points:
                        n_filtered += 1
                        continue
                    if max_total_points_per_example and (total_on + n > max_total_points_per_example):
                        if on:
                            sub_batches.append(on)
                            total_on = 0
                            on = []
                    on.append(ix)
                    total_on += n
                if on:
                    sub_batches.append(on)
                    
                    
                if max_samples_per_image is not None and len(sub_batches) > max_samples_per_image:
                    sub_batches = random.sample(sub_batches, max_samples_per_image)
                
                
                
                for ix in sub_batches:
                    flattened.append(dict(
                        ex,
                        label=[ex["label"][i] for i in ix],
                        points=[ex["points"][i] for i in ix],
                    ))
            logging.info(f"Filtered {n_filtered} ({n_filtered}/{total_points}) points")
            logging.info(f"Split {len(data)} examples into {len(flattened)} parts")
            self.data = flattened

    def __len__(self):
        if self.counting == "both":
            return len(self.data)*2
        else:
            return len(self.data)

    def get(self, item, rng):
        if self.counting == "both":
            mode = self.mode[item % 2]
            item = item // 2
        else:
            mode = self.mode

        ex = self.data[item]
        messages = []
        for label, points in zip(ex["label"], ex["points"]):
            messages.append(dict(
                label=label,
                points=np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1),
                point_scale=100,
                style=mode
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_url=ex["image_url"],
            )
        )


class PixMoPointExplanations(Dataset):

    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=1024, cache_only=False):
        local_name = join(PIXMO_DATASETS, "point-explanations")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-point-explanations", split="train")
        ds = ds.filter(lambda x: x is not None, input_columns=["parsed_response"])
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, split_groups=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.split_groups = split_groups
        data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "point-explanations"),
            keep_in_memory=keep_in_memory)[split]
        out = []
        for ex in data:
            sprvla_ex = dict(
                image=ex["image"],
                metadata=dict(
                    image_url=ex["image_url"],
                )
            )
            msg_list = []
            for q, res, alt, inline, points in zip(
                ex["question"], ex["parsed_response"],
                ex["alt_text"], ex["inline_text"], ex["points"]
            ):
                msg_list.append(dict(
                    question=q,
                    answer=res,
                    answer_annotations=[dict(
                        points=p, inline_text=i, alt_text=a
                    ) for p, i, a in zip(points, inline, alt)],
                    style="point_qa"
                ))
            if self.split_groups and len(msg_list) > 1:
                n = len(msg_list) // 2 + len(msg_list) % 2
                out.append(dict(sprvla_ex, message_list=msg_list[:n]))
                out.append(dict(sprvla_ex, message_list=msg_list[n:]))
            else:
                out.append(dict(sprvla_ex, message_list=msg_list))
        self.data = out

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        return dict(self.data[item])


class PixMoCapQa(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "cap-qa")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap-qa", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False, style="synthetic_qa"):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap-qa"), keep_in_memory=keep_in_memory)[split]
        self.style = style

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        messages = [dict(messages=msg, style=self.style) for msg in example["messages"]]

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_url=example["image_url"],
            )
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                messages = conv["messages"]
                for user_question_ix in range(0, len(messages), 2):
                    if re.fullmatch("how many.*", messages[user_question_ix].lower()):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        messages[user_question_ix] = prefix + messages[user_question_ix]
        return ex


class PixMoCap(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False, sample=None):
        local_name = join(PIXMO_DATASETS, "cap")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-cap", split="train")
        ds = add_internal_urls(ds)
        if sample:
            ds = ds.take(sample)
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        logging.info("Preparing data...")
        filtered_dataset = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        filtered_dataset = filtered_dataset.add_column(
            "image", [url_to_filename[x] for x in filtered_dataset["image_url"]])
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, mode, prefix_how_many=True, keep_in_memory=False, flatten=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        if mode not in ["transcript", "transcripts", "captions", "transcript_and_caption", "transcript1_and_caption"]:
            raise ValueError(mode)
        self.split = split
        self.mode = mode
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cap"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        messages = []
        caption = ex.pop("caption")
        transcripts = ex.pop("transcripts")
        if self.mode in ["captions", "transcript_and_caption", "transcript1_and_caption"]:
            messages.append(dict(text=caption, style="long_caption"))
        if self.mode in ["transcript_and_caption", "transcript1_and_caption", "transcript"]:
            if self.mode == "transcript_and_caption":
                ix = rng.randint(0, len(transcripts))
            else:
                ix = 0
            messages.append(dict(text=transcripts[ix], style="transcript"))
        if self.mode == "transcripts":
            messages += [dict(text=tr, style="transcript") for tr in transcripts]
        out = dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(
                image_path=ex["image"],
                image_url=ex.pop("image_url"),
            )
        )
        return out


class PixMoAskModelAnything(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, n_val=2048, cache_only=False):
        local_name = join(PIXMO_DATASETS, "ask-model-anything")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-ask-model-anything", split="train")
        filenames = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        filtered_dataset = filter_and_group_data(ds, filenames, check_sha)
        save_local_dataset(filtered_dataset, local_name, n_procs, n_val=n_val)

    def __init__(self, split, prefix_how_many=True, keep_in_memory=False):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "ask-model-anything"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        example = self.data[item]
        messages = []
        for q, a in zip(example["question"], example["answer"]):
            messages.append(dict(
                question=q,
                answer=a,
                style="user_qa"
            ))

        ex = dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_url=example["image_url"],
            )
        )

        if self.prefix_how_many:
            for conv in ex["message_list"]:
                if re.fullmatch("how many.*", conv["question"].lower()):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    conv["question"] = prefix + conv["question"]
        return ex


class PixMoPointsEval(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, cache_only=False):
        local_name = join(PIXMO_DATASETS, "pixmo-points-eval")
        if exists(local_name):
            return
        ds = datasets.load_dataset("allenai/pixmo-points-eval", split="test")
        url_to_filename = download_pixmo_urls(ds, n_procs, check_sha=check_sha, cache_only=cache_only, verify=VERIFY)
        ds = ds.filter(lambda x: x in url_to_filename, input_columns=["image_url"])
        ds = ds.add_column("image", [url_to_filename[x] for x in ds["image_url"]])
        save_local_dataset(ds, local_name, n_procs)

    def __init__(self, keep_in_memory=False):
        self.data = datasets.load_from_disk(
            join(PIXMO_DATASETS, "pixmo-points-eval"), keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        points = ex["points"]
        messages = []
        points = np.stack([[x["x"] for x in points], [x["y"] for x in points]], -1)
        return dict(
            image=ex["image"],
            label=ex["label"],
            points=points,
            point_scale=100,
            style="pointing",
            metadata=dict(
                points=points,
                masks=np.array(ex["masks"], dtype=bool),
                image_url=ex["image_url"],
            )
        )


class DenseCaptionEval(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __init__(self):
        with open(cached_path(join(PIXMO_DATASETS, "dense-caption-eval", "test.jsonl")), "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def get(self, item, rng):
        ex = json.loads(self.lines[item])
        return dict(
            image=join(DATA_HOME, "pixmo_images", ex["image"]),
            style="long_caption",
            metadata=dict(
                image_url=ex["url"],
            )
        )


class PixMoClocks(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError("Created from the original tfrecords")

    def __init__(self, split, aug=True):
        self.aug = aug
        super().__init__(split)

    def load(self):
        split = self.split
        src = join(PIXMO_DATASETS, "clocks", f"{split}.jsonl")
        logging.info(f"Loading pixmo clock data from {src}")
        with open(cached_path(src, cache_dir=os.environ.get("SPRVLA_CACHE_DIR"))) as f:
            return f.readlines()

    def get(self, item, rng: np.random.RandomState):
        ex = json.loads(self.data[item])

        time_format = ex["time_format"]
        shows_seconds = ex["shows_seconds"]
        hour, minute, second = [int(ex[k]) for k in ["hour", "minute", "second"]]
        if hour == 0:
            hour_str = "12"  # Midnight of the previous day
            am_pm = "AM"
        elif hour > 12:
            am_pm = "PM"
            hour_str = hour - 12
        else:
            hour_str = hour
            am_pm = "AM"
        hour_str = str(hour_str)
        minute_str = str(minute)
        if len(minute_str) == 1:
            minute_str = "0" + minute_str
        second_str = str(second)

        if len(second_str) == 1:
            second_str = "0" + second_str

        prefix = "The time shown is "
        if time_format == "The time is not shown":
            text = "The time is not shown in the image."
            hour, minute, second = -1, -1, -1
        else:
            if not shows_seconds:
                second = -1
            if time_format == "12 hour clock (without AM/PM)" and shows_seconds:
                if hour >= 12:
                    hour = hour - 12
                time = "".join([hour_str, ":", minute_str, ":", second_str])
            elif time_format == "12 hour clock (with AM/PM)" and shows_seconds:
                time = "".join([hour_str, ":", minute_str, ":", second_str, " ", am_pm])
            elif time_format == "12 hour clock (with AM/PM)" and not shows_seconds:
                time = "".join([hour_str, ":", minute_str, " ", am_pm])
            elif time_format == "12 hour clock (without AM/PM)" and not shows_seconds:
                if hour >= 12:
                    hour = hour - 12
                time = "".join([hour_str, ":", minute_str])
            else:
                raise RuntimeError()
            text = "".join(["The time shown is ", time])

        image = load_pil_image(join(PIXMO_DATASETS, "clocks", "images", ex["image"]))
        # Cutoff the black sharding at the bottom of every image
        image = image.crop((0, 0, image.width, image.height-120))

        if self.aug:
            sel = rng.random()
            if sel < 0.1:
                # Straight on
                shear_x = 0.
                shear_y = 0.
                rotation = 0.
            elif sel < 0.5:
                # Normal looking
                shear_x = rng.uniform(-10, 10)
                shear_y = rng.uniform(-10, 10)
                rotation = rng.uniform(-25, 25)
            else:
                if rng.random() > 0.5:
                    shear_x = rng.uniform( -30, 30)
                    shear_y = rng.uniform( -30, 30)
                else:
                    shear_x = rng.uniform( -10, 10)
                    shear_y = rng.uniform( -10, 10)
                rot_rng = rng.random()
                if rot_rng < 0.2:
                    rotation = rng.uniform( -25, 25)
                elif rot_rng < 0.6:
                    rotation = rng.uniform( -80, 80)
                else:
                    rotation = rng.uniform( -180, 180)

            if rng.random() > 0.5:
                scale = rng.uniform(0.3, 2)
            else:
                scale = rng.uniform(0.3, 1)

            # Avoid parts of the clock getting cutoff by the affine transform
            image = torchvision.transforms.Pad([200, 200, 200, 200], fill=255)(image)
            shear_y, shear_x = 0, 0
            image = affine(
                image,
                rotation,
                translate=[0, 0],
                scale=scale,
                shear=[shear_x, shear_y],
                interpolation=InterpolationMode.BILINEAR,
                fill=255
            )

            # Crop to whitespace
            bbox = ImageOps.invert(image).getbbox()
            image = image.crop(bbox)

            # Translate so the clock is not in the center
            height, width = image.height, image.width
            if rng.random() < 0.2:
                h_pad = rng.randint(0, height//2, (2,), dtype=np.int32)
                w_pad = rng.randint(0, width//2, (2,), dtype=np.int32)
            else:
                h_pad = rng.randint(0, height*2, (2,), dtype=np.int32)
                w_pad = rng.randint(0, width*2, (2,), dtype=np.int32)
            image = torchvision.transforms.Pad([h_pad[0], w_pad[0], h_pad[1], w_pad[1]], fill=255)(image)

            # Mild color jitter
            image = VF.adjust_hue(image, rng.uniform(-0.05, 0.05))
            image = VF.adjust_brightness(image, rng.uniform(0.85, 1.2))
            image = VF.adjust_saturation(image, rng.uniform(0.8, 1.2))
            image = VF.adjust_contrast(image, rng.uniform(0.8, 1.2))

        return dict(
            image=np.array(image),
            prompt="What time is being shown?",
            text=text,
            metadata=dict(hour=hour, second=second, minute=minute),
            style="clocks"
        )


class CoSyn(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        for name in [
            "chart", "chemical", "circuit", "diagram",
            "document", "graphic", "math", "music",
            "nutrition", "table"
        ]:
            local_name = join(PIXMO_DATASETS, f"cosyn-{name}")
            if exists(local_name):
                continue
            all_data = datasets.DatasetDict()
            for split in ["train", "validation"]:
                ds = datasets.load_dataset("allenai/CoSyn-400K", name=name, split=split)
                pil_images = (ex["image"] for ex in ds)
                filenames = [
                    join(COSYN_IMAGES, name, f"{img_id}.png")
                    for img_id in ds["id"]
                ]
                saved_images = save_images(pil_images, filenames, n_procs)
                assert len(saved_images) == len(filenames)
                def pil_to_path(ex):
                    ex["image"] = join(COSYN_IMAGES, name, f"{ex['id']}.png")
                    return ex
                new_features = ds.features.copy()
                new_features["image"] = datasets.Value("string")
                ds = ds.map(pil_to_path, features=new_features)
                all_data[split] = ds
            save_local_dataset(all_data, local_name, n_procs)
    
    def __init__(self, doc_type, split, use_exp=True, keep_in_memory=False):
        assert doc_type in [
            "chart", "chemical", "circuit", "diagram",
            "document", "graphic", "math", "music",
            "nutrition", "table"
        ]
        assert split in ["train", "validation"]
        self.doc_type = doc_type
        self.split = split
        self.use_exp = use_exp
        self.dataset = datasets.load_from_disk(
            join(PIXMO_DATASETS, f"cosyn-{doc_type}"), keep_in_memory=keep_in_memory)[split]
        # self.dataset = datasets.load_dataset(
        #     "allenai/CoSyn-400K", name=doc_type, split=split, keep_in_memory=keep_in_memory)

    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng):
        style = f"cosyn_{self.doc_type}"
        example = self.dataset[item]
        qeas = example["qa_pairs"]
        if self.use_exp:
            style += "_exp"
            message_list = [
                dict(question=q, explanation=e, answer=a, style=style) for q, e, a in
                zip(qeas["question"], qeas["explanation"], qeas["answer"])
            ]
        else:
            message_list = [
                dict(question=q, answer=a, style=style) for q, a in
                zip(qeas["question"], qeas["answer"])
            ]
        return dict(
            image=example["image"],
            message_list=message_list,
            metadata=dict(
                image_id=example["id"]
            )
        )


class CoSynPoint(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        local_name = join(PIXMO_DATASETS, "cosyn-point")
        if exists(local_name):
            return
        local_data_name = cached_path(
            join(PIXMO_DATASETS, "cosyn-point-data.json"), cache_dir=os.environ.get("SPRVLA_CACHE_DIR"),
        )
        with open(local_data_name, 'r') as f:
            data = json.load(f)
        id2data = {ex["id"]: ex for ex in data}
        all_data = datasets.DatasetDict()
        for split in ["train", "validation"]:
            ds = datasets.load_dataset("allenai/CoSyn-point", split=split)
            pil_images = (ex["image"] for ex in ds)
            filenames = [
                join(COSYN_IMAGES, "point", f"{img_id}.png")
                for img_id in ds["id"]
            ]
            saved_images = save_images(pil_images, filenames, n_procs)
            assert len(saved_images) == len(filenames)
            def pil_to_path(ex):
                ex["image"] = join(COSYN_IMAGES, "point", f"{ex['id']}.png")
                return ex
            new_features = ds.features.copy()
            new_features["image"] = datasets.Value("string")
            ds = ds.map(pil_to_path, features=new_features)
            ds = ds.add_column("names", [id2data[x]["names"] for x in ds["id"]])
            all_data[split] = ds
        save_local_dataset(all_data, local_name, n_procs)

    def __init__(self, split, keep_in_memory=False):
        assert split in ["train", "validation"]
        self.dataset = datasets.load_from_disk(
            join(PIXMO_DATASETS, "cosyn-point"), keep_in_memory=keep_in_memory)[split]

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = self.dataset[item]
        messages = []
        for question, points, name in zip(example["questions"], example["answer_points"], example["names"]):
            messages.append(dict(
                question=question,
                points=np.stack([points['x'], points['y']], -1),
                label=name,
                point_scale=100,
                style="cosyn_point",
            ))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(
                image_id=example["id"]
            )
        )


class CorrectionQa(Dataset):
    PREFIX = "https://explore-multimodal-datasets.s3.us-west-2.amazonaws.com/correction-urls"
    
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()
    
    def __init__(self, split, multi_image_only=False, max_images=None):
        assert split in ["train", "validation"]
        self.split = split
        self.max_images = max_images
        self.multi_image_only = multi_image_only
        self.dataset = self.load()
    
    def load(self):
        split = self.split
        local_data_name = cached_path(
            join(PIXMO_DATASETS, "correction-qa", f"{split}-records.json"),
            cache_dir=os.environ.get("SPRVLA_CACHE_DIR"),
        )
        with open(local_data_name, 'r') as f:
            records = json.load(f)
        
        group_by_images = defaultdict(list)
        for record in records:
            if "imageUrls" in record:
                group_by_images[tuple(record["imageUrls"])].append(record)
            else:
                group_by_images[tuple([record["imageUrl"]])].append(record)
        
        if self.multi_image_only:
            group_by_images = {k: v for k, v in group_by_images.items() if len(k) > 1}
        if self.max_images:
            group_by_images = {k: v for k, v in group_by_images.items() if len(k) <= self.max_images}

        out = []
        for image_urls, records in group_by_images.items():
            image = list(image_urls)
            questions = []
            answers = []
            for record in records:
                questions.append(record["question"])
                answers.append(record["answer"])
            out.append(
                dict(
                    image=image,
                    questions=questions,
                    answers=answers,
                    prolificId=[record["prolificId"] for record in records],
                    obejctID=[record["objectID"] for record in records],
                )
            )
        return out
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, item, rng: np.random.RandomState):
        example = self.dataset[item]
        image = example["image"]
        dst_prefix = join(DATA_HOME, "correction_images")
        if len(image) > 1:
            image = [url.replace(self.PREFIX, dst_prefix) for url in image]
        else:
            image = image[0].replace(self.PREFIX, dst_prefix)
        messages = []
        for q, a in zip(example["questions"], example["answers"]):
            messages.append(dict(question=q, answer=a, style="correction_qa"))
        rng.shuffle(messages)
        return dict(
            image=image,
            message_list=messages,
        )
