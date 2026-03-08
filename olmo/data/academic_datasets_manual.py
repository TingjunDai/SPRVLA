"""Datasets the load directly from source files,
Currently not used in favour of using HF datasets"""
import json
import logging
from os import environ
from os.path import exists
from os.path import join
import numpy as np
import copy
import pandas as pd
import io
from PIL import Image
from collections import defaultdict
from cached_path import cached_path

from olmo.data.dataset import DATA_HOME, DatasetBase, Dataset
from olmo.io import read_file, write_json, file_exists


if DATA_HOME is not None:
    DOWNLOADS = join(DATA_HOME, "downloads")
    CHARTQA_SOURCE = join(DATA_HOME, "chartqa")
    DOCQA_SOURCE = join(DATA_HOME, "docqa")
    INFOQA_SOURCE = join(DATA_HOME, "info_qa")
    ST_QA_SRC = join(DATA_HOME, "scene-text")
    VQA2_SOURCE = join(DATA_HOME, "vqa2")
    A_OKVQA_SOURCE = join(DATA_HOME, "a_okvqa")
    TEXT_VQA_SOURCE = join(DATA_HOME, "text_vqa")
    TALLY_QA_SOURCE = join(DATA_HOME, "tally_qa")
    FIGURE_QA_SOURCE = join(DATA_HOME, "figure_qa")
    PLOT_QA_SOURCE = join(DATA_HOME, "plot_qa")
    DVQA_SOURCE = join(DATA_HOME, "dvqa")
else:
    CHARTQA_SOURCE = None
    DOCQA_SOURCE = None
    INFOQA_SOURCE = None
    ST_QA_SRC = None
    VQA2_SOURCE = None
    A_OKVQA_SOURCE = None
    TEXT_VQA_SOURCE = None
    TALLY_QA_SOURCE = None
    FIGURE_QA_SOURCE = None
    PLOT_QA_SOURCE = None
    DVQA_SOURCE = None


class Vqa2(Dataset):
    SPLITS = ["train", "val", "test"]
    
    def __init__(self, split, multi_question=False, sample=None):
        split = "val" if split == "validation" else split
        assert split in self.SPLITS
        self.split = split
        self.multi_question = multi_question
        self.dataset = self.load()
        if not self.multi_question:
            flattened_data = []
            for item in self.dataset:
                for q in item["messages"]:
                    flattened_data.append(dict(
                        style=q['style'],
                        question=q["question"],
                        answers=q["answers"],
                        image=item["image"],
                        image_id=item["image_id"],
                        question_id=q["question_id"],
                    ))
            if sample:
                logging.info(f"Sampling {sample} of {len(flattened_data)} ({100*sample/len(flattened_data)}:0.1f)")
                np.random.RandomState(9123).shuffle(flattened_data)
                flattened_data = flattened_data[:sample]
            self.dataset = flattened_data
        else:
            assert sample is None

    def __len__(self):
        return len(self.dataset)
    
    def load(self):
        split = self.split
        if file_exists(join(VQA2_SOURCE, f"sprvla_{split}.json")):
            return json.loads(read_file(join(VQA2_SOURCE, f"sprvla_{split}.json")))

        if split == "test":
            q_filename = "v2_OpenEnded_mscoco_test2015_questions.json"
            image_dir = "test2015"
        else:
            q_filename = f"v2_OpenEnded_mscoco_{split}2014_questions.json"
            image_dir = f"{split}2014"
        q_filename = join(VQA2_SOURCE, q_filename)
        questions = json.loads(read_file(q_filename))

        a_filename = join(VQA2_SOURCE, f"v2_mscoco_{split}2014_annotations.json") if split != "test" else None
        if a_filename is not None:
            annotations = json.loads(read_file(a_filename))
            qa = {ann["question_id"]: ann for ann in annotations["annotations"]}
        else:
            qa = None
        
        grouped_by_image = defaultdict(list)
        for q in questions["questions"]:
            grouped_by_image[q["image_id"]].append(q)
        
        out = []
        image_ids = sorted(grouped_by_image.keys())
        for image_id in image_ids:
            questions = grouped_by_image[image_id]
            messages = []
            for question in questions:
                anno = qa[question["question_id"]] if qa is not None else None
                answers = [x["answer"] for x in anno["answers"]] if anno is not None else None
                messages.append(
                    dict(
                        question=question["question"],
                        answers=answers,
                        question_id=question["question_id"],
                        style="vqa2",
                    )
                )
            out.append(
                dict(
                    messages=messages,
                    image_id=image_id,
                    image=join(VQA2_SOURCE, image_dir, f"COCO_{image_dir}_{image_id:0>12}.jpg"),
                )
            )
        write_json(join(VQA2_SOURCE, f"sprvla_{split}.json"), out)
        return out

    def get(self, item, rng):
        ex = self.dataset[item]
        if self.multi_question:
            return dict(
                metadata=dict(image_id=ex["image_id"]),
                image=ex["image"],
                message_list=ex["messages"],
            )
        else:
            return dict(
                style="vqa2",
                answers=ex["answers"],
                metadata=dict(image_id=ex["image_id"], example_id=ex["question_id"]),
                image=ex["image"],
                question=ex["question"],
            )


class AOkVqa(Dataset):
    SPLITS = ["train", "val", "test"]


    def __init__(self, split, direct_answer=False):
        split = "val" if split == "validation" else split
        assert split in self.SPLITS
        self.split = split
        self.direct_answer = direct_answer
        self.style = "a_okvqa_" + ("da" if direct_answer else "mc")
        self.loaded_data = self.load()
    
    def load(self):
        split = self.split
        a_filename = join(A_OKVQA_SOURCE, f"aokvqa_v1p0_{split}.json")
        data = json.loads(read_file(a_filename))
        
        loaded_data = []
        for ex in data:
            image_id = ex["image_id"]
            image_file_name = join(A_OKVQA_SOURCE, f"{split}2017", f"{image_id:0>12}.jpg")

            if self.direct_answer:
                if ex["difficult_direct_answer"] and self.split in ["val", "test"]:
                    continue
                out = dict(
                    image=image_file_name,
                    question=ex["question"],
                    answers=ex.get("direct_answers", None),
                    metadata=dict(
                        example_id=ex["question_id"]
                    )
                )
            else:
                correct_choice_idx = ex.get('correct_choice_idx', None)
                if correct_choice_idx is None:
                    out = dict(
                        image=image_file_name,
                        question=ex["question"],
                        options=ex["choices"],
                        metadata=dict(
                            example_id=ex["question_id"]
                        )
                    )
                else:
                    out = dict(
                        image=image_file_name,
                        question=ex["question"],
                        options=ex["choices"],
                        answer_idx=correct_choice_idx,
                        metadata=dict(
                            example_id=ex["question_id"]
                        )
                    )
            loaded_data.append(out)
        return loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def get(self, item, rng):
        return dict(**self.loaded_data[item], style=self.style)


class TextVqa(Dataset):
    SPLITS = ["train", "val", "test"]
    _NUM_ANSWERS_PER_QUESTION = 10

    def __init__(self, split: str, identifier=None):
        split = "val" if split == "validation" else split
        assert split in self.SPLITS
        self.split = split
        self.identifier = identifier
        self.dataset = self.load()
    
    def load(self):
        data = json.loads(read_file(join(TEXT_VQA_SOURCE, f"TextVQA_0.5.1_{self.split}.json")))['data']
        out = []
        for item in data:
            item = copy.deepcopy(item)
            item["answers"] = item.get("answers", ["" for _ in range(self._NUM_ANSWERS_PER_QUESTION)])
            image_id = item["image_id"]
            image_subfolder = "train_images" if item["set_name"] != "test" else "test_images"
            item["image"] = join(TEXT_VQA_SOURCE, image_subfolder, f"{image_id}.jpg")
            out.append(item)
        return out

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        example = self.dataset[item]
        return dict(
            image=example["image"],
            question=example["question"],
            answers=example.get("answers", []),
            metadata=dict(
                image_url=example["flickr_300k_url"],
                image_id=example["image_id"],
                example_id=example["question_id"],
            ),
            style="text_vqa"
        )


class TallyQa(Dataset):
    SPLITS = ["train", "test"]
    IMAGE_SOURCES = {
        "train2014": VQA2_SOURCE,
        "val2014": VQA2_SOURCE,
        "VG_100K": TALLY_QA_SOURCE,
        "VG_100K_2": TALLY_QA_SOURCE
    }

    def __init__(self, split):
        assert split in self.SPLITS
        self.split = split
        self.dataset = self.load()
    
    def load(self):
        data = json.loads(read_file(join(TALLY_QA_SOURCE, f"{self.split}.json")))
        grouped_by_image = defaultdict(list)
        for ex in data:
            grouped_by_image[ex["image"]].append(ex)

        out = []
        for image, questions in grouped_by_image.items():
            image_id = questions[0]["image_id"]
            for q in questions:
                assert q.pop("image_id") == image_id
                assert q.pop("image") == image
                if "issimple" in q:
                    q["issimple"] = int(q["issimple"])
                else:
                    q["issimple"] = -1
            image_src, path = image.split("/")
            image_path = join(self.IMAGE_SOURCES[image_src], image_src, path)
            record = dict(
                image_id=image_id,
                image=image_path,
                questions=questions,
            )
            record["image/filename"] = image
            out.append(record)
        return out

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        messages = []
        questions = [q["question"] for q in ex["questions"]]
        answers = [q["answer"] for q in ex["questions"]]
        for question, answer in zip(questions, answers):
            messages.append(dict(
                question=question,
                answer=str(answer),
                style="tally_qa"
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict(image_id=ex["image_id"])
        )


class InfoQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                filename = "infographicsVQA_val_v1.0_withQT.json"
            else:
                filename = f"infographicsVQA_{split}_v1.0.json"
            if not exists(join(INFOQA_SOURCE, filename)):
                raise ValueError(
                    "InfoQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 3)"
                    f" please download and unzip the data into `{INFOQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            filename = "infographicsVQA_val_v1.0_withQT.json"
        else:
            filename = f"infographicsVQA_{split}_v1.0.json"
        filename = join(INFOQA_SOURCE, filename)
        logging.info(f"Loading infoqa data from {filename}")
        with open(cached_path(filename, cache_dir=environ.get("SPRVLA_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            image_path = join(INFOQA_SOURCE, "images", ex.pop("image_local_name"))
            out.append(dict(
                image=image_path,
                question=ex["question"],
                answers=ex.get("answers", []),
                metadata=dict(example_id=ex["questionId"]),
            ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style="info_qa")


class DocQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                split = "val"
            if split == "test":
                src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
            else:
                src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
            if not exists(src):
                raise ValueError(
                    "DocQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 1)"
                    f" please download and unzip the data into `{DOCQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        if self.split == "test":
            src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
        else:
            src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
        logging.info(f"Loading docqa data from {src}")
        with open(cached_path(src, cache_dir=environ.get("SPRVLA_CACHE_DIR"))) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            assert ex.pop("data_split") == split
            image_path = join(DOCQA_SOURCE, ex["image"])
            if self.split == "test":
                for k in ["answers", "question_types"]:
                    assert k not in ex
                    ex[k] = []
            out.append(dict(
                image=join(DOCQA_SOURCE, ex["image"]),
                question=ex["question"],
                answers=ex.get("answers"),
                metadata=dict(
                    doc_id=ex["docId"],
                    question_types=ex.get("question_types"),
                    example_id=ex["questionId"],
                ),
            ))
        return out

    def get(self, item, rng):
        return dict(self.data[item], style="doc_qa")


class ChartQa(DatasetBase):
    def __init__(self, split, parts="both", weighted=False, use_exp=False):
        self.weighted = weighted
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]
        self.parts = parts
        self.use_exp = use_exp
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        examples = []
        if self.parts == "both":
            parts = ["human", "augmented"]
        else:
            parts = [self.parts]
        for part in parts:
            src = f"{CHARTQA_SOURCE}/{split}/{split}_{part}.json"
            logging.info(f"Loading chartqa data from {src}")
            with open(cached_path(src, cache_dir=environ.get("SPRVLA_CACHE_DIR"))) as f:
                data = json.load(f)
            for ex_id, ex in enumerate(data):
                ex = dict(
                    image=join(CHARTQA_SOURCE, split, "png", ex.pop("imgname")),
                    question=ex["query"],
                    answers=ex["label"],
                    metadata=dict(
                        is_human=part == "human",
                        example_id=ex_id
                    )
                )
                examples.append(ex)
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="chart_qa_exp" if self.use_exp else "chart_qa")
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex


class FigureQa(Dataset):

    def __init__(self, split):
        assert split in ["train", "validation1", "test1", "validation2", "test2"]
        self.split = split
        self.hf_dataset = self.load()
    
    def load(self):
        split = self.split
        if file_exists(join(FIGURE_QA_SOURCE, f"sprvla_{split}.json")):
            return json.loads(read_file(join(FIGURE_QA_SOURCE, f"sprvla_{split}.json")))

        if split == "train":
            folder_name = "train1"
        elif "test" in split:
            folder_name = "no_annot_" + split
        else:
            folder_name = split
        source_dir = join(FIGURE_QA_SOURCE, folder_name)
        data = json.loads(read_file(join(source_dir, "qa_pairs.json")))
        grouped_by_image = defaultdict(list)
        for question in data["qa_pairs"]:
            grouped_by_image[question["image_index"]].append(question)
        out = []
        for image_index, questions_data in grouped_by_image.items():
            questions = []
            answers = []
            for q in questions_data:
                questions.append(q["question_string"])
                answers.append(q.get("answer"))
            out.append(
                dict(
                    image=join(source_dir, "png", str(image_index) + ".png"),
                    image_index=image_index,
                    questions=questions,
                    answers=answers,
                )
            )
        write_json(join(FIGURE_QA_SOURCE, f"sprvla_{split}.json"), out)
        return out

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        messages = []
        for q, a in zip(example["questions"], example["answers"]):
            messages.append(dict(question=q, answer=str(a), style="figure_qa"))
        return dict(image=example["image"], message_list=messages)


class PlotQa(Dataset):
    SPLITS = ["train", "validation", "test"]

    def __init__(self, split):
        assert split in self.SPLITS
        self.split = split
        self.hf_dataset = self.load()
    
    def load(self):
        split = self.split
        if file_exists(join(PLOT_QA_SOURCE, f"sprvla_{split}.json")):
            return json.loads(read_file(join(PLOT_QA_SOURCE, f"sprvla_{split}.json")))

        data = json.loads(read_file(join(PLOT_QA_SOURCE, f"{split}_qa_pairs_V2.json")))
        grouped_by_image = defaultdict(list)
        for question in data["qa_pairs"]:
            grouped_by_image[question["image_index"]].append(question)
        out = []
        for image_index, questions_data in grouped_by_image.items():
            questions = []
            answers = []
            for q in questions_data:
                questions.append(q["question_string"])
                answers.append(str(q.get("answer")))
            out.append(
                dict(
                    image=join(PLOT_QA_SOURCE, f"{split}_png", str(image_index) + ".png"),
                    image_index=image_index,
                    questions=questions,
                    answers=answers,
                )
            )
        write_json(join(PLOT_QA_SOURCE, f"sprvla_{split}.json"), out)
        return out

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        messages = []
        for q, a in zip(example["questions"], example["answers"]):
            messages.append(dict(question=q, answer=a, style="plot_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class DvQa(Dataset):
    SPLITS = ["train", "val_hard", "val_easy"]

    def __init__(self, split):
        assert split in self.SPLITS
        self.split = split
        self.hf_dataset = self.load()
    
    def load(self):
        split = self.split
        if file_exists(join(DVQA_SOURCE, f"sprvla_{split}.json")):
            return json.loads(read_file(join(DVQA_SOURCE, f"sprvla_{split}.json")))

        data = json.loads(read_file(join(DVQA_SOURCE, f"{split}_qa.json")))
        grouped_by_image = defaultdict(list)
        for question in data:
            grouped_by_image[question["image"]].append(question)
        out = []
        for image, questions_data in grouped_by_image.items():
            questions = [q["question"] for q in questions_data]
            answers = [q["answer"] for q in questions_data]
            out.append(
                dict(
                    image_id=image,
                    image=join(DVQA_SOURCE, "images", image),
                    questions=questions,
                    answers=answers,
                )
            )
        write_json(join(DVQA_SOURCE, f"sprvla_{split}.json"), out)
        return out

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        messages = []
        for q, a in zip(example["questions"], example["answers"]):
            messages.append(dict(question=q, answer=a, style="dv_qa"))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(image_id=example["image_id"]),
        )


class SceneTextQa(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        for split in ["train", "test"]:
            if not exists(join(join(ST_QA_SRC, f"{split}_task_3.json"))):
                raise ValueError(
                    "SceneTextQa requires manually downloading https://rrc.cvc.uab.es/?ch=11"
                    f" please download and unzip the data into `{ST_QA_SRC}`"
                )

    def __init__(self, split):
        assert split in ["train", "test", "validation"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "train"
        src = join(ST_QA_SRC, f"{self.split}_task_3.json")
        logging.info(f"Loading scene text data from {src}")
        data = json.loads(read_file(src))["data"]
        out = []
        for question in data:
            out.append(dict(
                image=join(ST_QA_SRC, question["file_path"]),
                question=question["question"],
                metadata=dict(example_id=question["question_id"]),
                answers=question.get("answers", []),
            ))
        if self.split in ["train", "validation"]:
            # Custom val split since the data doesn't have one
            out.sort(key=lambda x: x["metadata"]["example_id"])
            np.random.RandomState(63069).shuffle(out)
            if self.split == "train":
                return out[1024:]
            else:
                return out[:1024]
        else:
            return out

    def get(self, item, rng):
        return dict(self.data[item], style="st_qa")