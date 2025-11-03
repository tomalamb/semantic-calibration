# Copyright (C) 2023-24 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance w
# ith the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convenience wrappers around classification datasets
"""
import torch as t

from abc import abstractmethod
from enum import Enum
from transformers import AutoTokenizer
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import datasets
from datasets import load_dataset
import re
import string
import random
from sklearn.model_selection import train_test_split

# List of datasets available in this module
dsets = [
    "boolq",
    "obqa",
    "arc",
    "winogrande",
    "cqa",
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp ",
    "rte ",
    "sst2",
    "wnli",
    "trivia_qa",
    "natural_questions",
    "squad",
    "coqa",
    "red_trivia_qa",
    "red_triiva_qa_ext",
    "gsm8k"
    "ambig_qa",
]


class ClassificationDataset:
    """
    An abstract base dataset for sequence classification problems. Multiple
    choice QA problems could also be made a subclass of this class with an
    appropriate collation/formatting.
    """

    def __init__(
        self,
        dset,
        tokenizer,
        n_labels: int,
        preamble: str = "",
        add_space: bool = False,
        numerical: bool = True,
        boolean: bool = False,
        few_shot: bool = False,
        max_len: int = 1024,
        # New parameters for three-way splitting:
        train_split_ratios: tuple = (0.9, 0.025, 0.075),  # (training, early_stopping, validation)
        calib_split_ratios: tuple = (0.8, 0.05, 0.15),  # (calib_training, calib_early_stopping, calib_validation)
        seed: int = 42,
    ):
        self.dset = dset
        self.n_labels = n_labels
        self.preamble = preamble
        self.add_space = add_space
        self.tokenizer = tokenizer
        self.numerical = numerical
        self.few_shot = few_shot
        self.max_len = max_len
        self.seed = seed
        spc = " " if self.add_space else ""

        # Initialize labels and mappings
        if numerical and boolean:
            raise ValueError("Question type cannot be both numerical and boolean")
        if boolean:
            labels = [f"{spc}True", f"{spc}False"]
        elif numerical:
            labels = [f"{spc}{i}" for i in range(self.n_labels)]
        else:  # alphabetical
            labels = [f"{spc}{chr(ord('A') + i)}" for i in range(self.n_labels)]
        self.target_ids = tokenizer(
            labels, return_tensors="pt", add_special_tokens=False
        ).input_ids[:, -1:]
        assert (
            self.target_ids.unique().numel() == self.target_ids.numel()
        ), "Target label IDS are not unique! Try changing add_space or numerical."

        self.label_idx2target_id = OrderedDict(
            [(i, self.target_ids[i]) for i in range(n_labels)]
        )
        self.target_id2label_idx = OrderedDict(
            [(self.target_ids[i], i) for i in range(n_labels)]
        )

        # Prepare splits using the new paradigm.
        self.prepare_splits(train_split_ratios, calib_split_ratios)

    def prepare_splits(self, train_split_ratios, calib_split_ratios):
        """
        Splits the original training set into two groups:
        - Training Group: 75% of self.dset["train"]
        - Calibration Group: 25% of self.dset["train"]
        
        Each group is then further split into three subsets using the provided ratios:
        For the training group (using train_split_ratios):
            - training_split, early_stopping_split, validation_split
        For the calibration group (using calib_split_ratios):
            - calib_training_split, calib_early_stopping_split, calib_validation_split
        
        The final test split is taken from self.dset["test"] (if available) or from
        self.dset["validation"] (capped to 4000 examples).
        """
        # ----- Split original training data into Training and Calibration groups -----
        train_data = self.dset["train"].shuffle(seed=self.seed)
        total_train = len(train_data)
        n_train_group = int(0.75 * total_train)
        n_calib_group = total_train - n_train_group
        
        train_group = train_data.select(range(n_train_group))
        calib_group = train_data.select(range(n_train_group, total_train))
        
        # ----- Further split the Training Group using train_split_ratios (3-tuple) -----
        tr_ratio, es_ratio, val_ratio = train_split_ratios
        if not abs(tr_ratio + es_ratio + val_ratio - 1.0) < 1e-6:
            raise ValueError("Training split ratios must sum to 1.")
        
        n_tr = int(tr_ratio * n_train_group)
        n_es = int(es_ratio * n_train_group)
        n_val = n_train_group - n_tr - n_es
        
        self.training_split = train_group.select(range(n_tr))
        self.early_stopping_split = train_group.select(range(n_tr, n_tr + n_es))
        self.validation_split = train_group.select(range(n_tr + n_es, n_train_group))
        
        # ----- Further split the Calibration Group using calib_split_ratios (3-tuple) -----
        c_tr_ratio, c_es_ratio, c_val_ratio = calib_split_ratios
        if not abs(c_tr_ratio + c_es_ratio + c_val_ratio - 1.0) < 1e-6:
            raise ValueError("Calibration split ratios must sum to 1.")
        
        n_ct = int(c_tr_ratio * n_calib_group)
        n_ces = int(c_es_ratio * n_calib_group)
        n_c_val = n_calib_group - n_ct - n_ces
        
        self.calib_training_split = calib_group.select(range(n_ct))
        self.calib_early_stopping_split = calib_group.select(range(n_ct, n_ct + n_ces))
        self.calib_validation_split = calib_group.select(range(n_ct + n_ces, n_calib_group))
        
        # Cap calibration validation split to 2000 examples.
        self.calib_validation_split = self.apply_subset_limit(
            self.calib_validation_split, max_samples=2000, seed=self.seed
        )
            
        # ----- Final Test Split -----
        if "test" in self.dset:
            self.final_test_split = self.apply_subset_limit(
                self.dset["test"], max_samples=2000, seed=self.seed
            )

        else:
            self.final_test_split = self.apply_subset_limit(
                self.dset["validation"], max_samples=2000, seed=self.seed
            )

    def apply_subset_limit(self, dset, max_samples=8000, seed=42):
        """Limit dataset to a maximum of max_samples examples."""
        dset = dset.shuffle(seed=seed)
        if len(dset) > max_samples:
            return dset.select(range(max_samples))
        return dset

    def sc_loader(self, subset, *args, **kwargs) -> DataLoader:
        return t.utils.data.DataLoader(
            subset, collate_fn=self.sc_collate_fn, *args, **kwargs
        )

    def clm_loader(self, subset, *args, **kwargs) -> DataLoader:
        return t.utils.data.DataLoader(
            subset, collate_fn=self.clm_collate_fn, *args, **kwargs
        )

    def loader(self, *args, is_sc: bool = False, with_correctness: bool = False, **kwargs):
        """
        Returns DataLoaders for the following splits:
        - training, early_stopping, validation (from the Training Group)
        - calib_training, calib_early_stopping, calib_validation (from the Calibration Group)
        - final_test (from the original validation or test split)
        """
        splits = {
            "training": self.training_split,
            "early_stopping": self.early_stopping_split,
            "validation": self.validation_split,
            "calib_training": self.calib_training_split,
            "calib_early_stopping": self.calib_early_stopping_split,
            "calib_validation": self.calib_validation_split,
            "final_test": self.final_test_split,
        }
        for name, subset in splits.items():
            print(f"{name} subset size: {len(subset)}")
        
        if is_sc:
            if with_correctness:
                collate_fn = self.sc_collate_fn_with_correctness
            else:
                collate_fn = self.sc_collate_fn
        else:
            collate_fn = self.clm_collate_fn
        loaders = {
            name: t.utils.data.DataLoader(subset, collate_fn=collate_fn, *args, **kwargs)
            for name, subset in splits.items()
        }
        return loaders

    def sc_collate_fn(self, batch):
        # Placeholder: to be implemented in subclasses.
        raise NotImplementedError

    def clm_collate_fn(self, batch):
        # Placeholder: to be implemented in subclasses.
        raise NotImplementedError
    
    def sc_collate_fn_with_correctness(self, batch):
        # First call the normal sc_collate_fn to get (prompts, classes, normalized_answers)
        prompts, classes, normalized_answers = self.sc_collate_fn(batch)
        
        # Then gather the is_correct field
        is_correct_list = []
        for ex in batch:
            # default is_correct to 0 if missing
            is_correct_list.append(ex.get("is_correct", 0))
        
        # Convert to a tensor
        is_correct_tensor = t.tensor(is_correct_list, dtype=t.int64)
        
        # Return a 4-tuple
        return prompts, classes, normalized_answers, is_correct_tensor



class BoolQDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 256,
    ):
        dset = load_dataset("boolq")

        prompt = """Read the passage below and answer the question with the words 'True' or 'False'.

Passage: {passage}
Question: {question}
Answer (True or False):"""

        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            boolean=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    def clm_collate_fn(self, batch):
        prompts = [
            self.preamble.format(
                passage=e["passage"][-self.max_len:], question=e["question"]
            )
            for e in batch
        ]
        classes = t.tensor([int(e["answer"]) for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [
            self.preamble.format(
                passage=e["passage"][-self.max_len:], question=e["question"]
            )
            for e in batch
        ]
        classes = t.tensor([int(e["answer"]) for e in batch])
        return prompts, classes, None


boolq = BoolQDataset


class OBQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 1024,
    ):
        dset = load_dataset("openbookqa", "main")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            4,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

The sun is responsible for
Choices:
A) puppies learning new tricks
B) children growing up and getting old
C) flowers wilting in a vase
D) plants sprouting, blooming and wilting
Answer: D

What doesn't eliminate waste?
A) plants
B) robots
C) mushrooms
D) bacteria
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Chioces:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for c, l, in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(
                    question=e["question_stem"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        return prompts, classes, None


obqa = OBQADataset


class ArcSplit(Enum):
    C = "ARC-Challenge"
    E = "ARC-Easy"


class ARCDataset(ClassificationDataset):
    """Modified the code here to get the model to predict the text for the right answer rather than just the label."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: ArcSplit = ArcSplit.E,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):

        dset = load_dataset("ai2_arc", name.value)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            5,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

        self.numeric_to_alpha = {
            "1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    few_shot_preamble = """Answer each question by writing the text of the choice that correctly answers the question. Do not use letters to indicate your answer.

Which two body systems are directly involved in movement?
Choices:
A) muscular and skeletal
B) digestive and muscular
C) skeletal and respiratory
D) respiratory and digestive
Answer: muscular and skeletal

Which of these is not an inherited trait in humans?
A) height
B) hair color
C) skin color
D) intelligence
Answer: intelligence

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Answer each question by writing the text of the choice that correctly answers the question. Do not use letters to indicate your answer.

Question: {question}
Choices:
{choices}
Answer:"""

    def _convert_answer_key(self, answer_key):
        if answer_key in self.numeric_to_alpha:
            return self.numeric_to_alpha[answer_key]
        return answer_key

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{self._convert_answer_key(l)}) {c}"
                    for c, l in zip(e["choices"]["text"], e["choices"]["label"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = []
        for e in batch:
            answer_key = self._convert_answer_key(e["answerKey"])
            answer_index = ord(answer_key) - ord("A")
            correct_answers.append(+e["choices"]["text"][answer_index])

        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}

        return prompts, correct_answers, None

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        correct_answers = []
        for e in batch:
            answer_key = self._convert_answer_key(e["answerKey"])
            answer_index = ord(answer_key) - ord("A")
            correct_answers.append(e["choices"]["text"][answer_index])

        return prompts, correct_answers, None


arc = ARCDataset


class WinograndeSplit(Enum):
    XS = "winogrande_xs"
    S = "winogrande_s"
    M = "winogrande_m"
    L = "winogrande_l"
    XL = "winogrande_xl"


class WinograndeDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        name: WinograndeSplit = WinograndeSplit.S,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("winogrande", name.value)
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Return the label of the correct answer for each question below.

Adam put handwash only clothes in the washer but Aaron washed them by hand as _ was lazy.
Choices:
A) Adam
B) Aaron
Answer: A

Steven proudly showed Michael the mangoes he grew himself all this summer. _ is astonished.
Choices:
A) Stephen
B) Michael
Answer: B

{question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Return the label of the correct answer for the question below.

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = f"A) {e['option1']}\nB) {e['option2']}"
            prompts.append(
                self.preamble.format(question=e["sentence"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([int(e["answer"]) - 1 for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([int(e["answer"]) - 1 for e in batch])
        return prompts, classes, None


winogrande = WinograndeDataset


class CommonsenseQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = True,
        max_len=4096,
    ):
        dset = load_dataset("commonsense_qa")
        super().__init__(
            dset,
            tokenizer,
            5,
            self.few_shot_preamble if few_shot else self.zero_shot_preamble,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    # few-shot preamble
    few_shot_preamble = """Answer the questions below correctly.

Question: What do people aim to do at work?
Choices:
A) complete job
B) learn from each other
C) kill animals
D) wear hats
E) talk to each other
Answer: A

Question: Where do adults use glue sticks?
Choices:
A) classroom
B) desk drawer
C) at school
D) office
E) kitchen draw
Answer: D

Question: {question}
Choices:
{choices}
Answer:"""

    zero_shot_preamble = """Answer the multiple choice question below by returning the answer label (A to E)

Question: {question}
Choices:
{choices}
Answer:"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            choices = "\n".join(
                [
                    f"{l}) {c}"
                    for l, c in zip(e["choices"]["label"], e["choices"]["text"])
                ]
            )
            prompts.append(
                self.preamble.format(question=e["question"], choices=choices)
            )
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        targets = t.cat([self.label_idx2target_id[c.item()] for c in classes])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([ord(e["answerKey"]) - ord("A") for e in batch])
        return prompts, classes, None


cqa = CommonsenseQADataset


class CoLADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "cola")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether it is grammatically acceptable (1) or unacceptable (0).

Sentence: If you had eaten more, you would want less.
Answer: 1

Sentence: As you eat the most, you want the least.
Answer: 0

Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether it is grammatically acceptable (1) or unacceptable (0).

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


cola = CoLADataset


class MNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "mnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            3,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each premise below, indicate whether the hypothesis entails (0), is neutral towards (1) or contradicts (2) the premise.

Hypothesis: Buffet and a la carte available.
Premise: It has a buffet.
Answer: 0

Hypothesis: He had never felt better.
Premise: The medicine he had taken had worked well.
Answer: 1

Hypothesis: Oh, what a fool I feel!
Premise: I am beyond proud
Answer: 2

Hypothesis: {hypothesis}
Premise: {premise}
Answer:"""

    zero_shot_preamble = """For each premise below, indicate whether the hypothesis entails (0), is neutral towards (1) or contradicts (2) the premise.

Hypothesis: Buffet and a la carte available.
Premise: It has a buffet.
Answer: 0

Hypothesis: He had never felt better.
Premise: The medicine he had taken had worked well.
Answer: 1

Hypothesis: Oh, what a fool I feel!
Premise: I am beyond proud
Answer: 2

Hypothesis: {hypothesis}
Premise: {premise}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                hypothesis=e["hypothesis"], premise=e["premise"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["hypothesis"] + " " + e["premise"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


mnli = MNLIDataset


class MRPCDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "mrpc")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the Sentence 1 is equivalent (1) or not equivalent (2) to the Sentence 2.

Sentence 1: Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.
Sentence 2: Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998.
Answer: 0

Sentence 1: Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.
Sentence 2: Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the Sentence 1 is equivalent (1) or not equivalent (2) to the Sentence 2.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


mrpc = MRPCDataset


class QNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "qnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether it entails (0) or does not entail (1) the associated question.

Question: Which collection of minor poems are sometimes attributed to Virgil?
Sentence: A number of minor poems, collected in the Appendix Vergiliana, are sometimes attributed to him.
Answer: 0

Question: What was the highest order of species n land?
Sentence: The climate was much more humid than the Triassic, and as a result, the world was very tropical.
Answer: 1

Question: {question}
Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether it entails (0) or does not entail (1) the associated question.

Question: {question}
Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                question=e["question"], sentence=e["sentence"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["question"] + " " + e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


qnli = QNLIDataset


class QQPDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "qqp")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of questions below, indicate whether the first is a duplicate (1) or not a duplicate (0) of the first.

Question 1: How is air traffic controlled?
Question 2: How do you become an air traffic controller?
Answer: 0

Question 1: What are the coolest Android hacks and tricks you know?
Question 2: What are some cool hacks for Android phones?
Answer: 1

Question 1: {question_1}
Question 2: {question_2}
Answer:"""

    zero_shot_preamble = """For each pair of questions below, indicate whether the first is a duplicate (1) or not a duplicate (0) of the first.

Question 1: {question_1}
Question 2: {question_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                question_1=e["question1"], question_2=e["question2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["question1"] + " " + e["question2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


qqp = QQPDataset


class RTEDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "rte")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the second entails (0) or does not entail (1) the first.

Sentence 1: Edward VIII became King in January of 1936 and abdicated in December.
Sentence 2: King Edward VIII abdicated in December 1936.
Answer: 0

Sentence 1: No Weapons of Mass Destruction Found in Iraq Yet.
Sentence 2: Weapons of Mass Destruction Found in Iraq.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the second entails (0) or does not entail (1) the first.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


rte = RTEDataset


class SST2Dataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "sst2")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=True,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each sentence below, indicate whether the sentiment is negative (0) or positive (1).

Sentence: a depressed fifteen-year-old 's suicidal poetry
Answer: 0

Sentence: the greatest musicians
Answer: 1

Sentence: {sentence}
Answer:"""

    zero_shot_preamble = """For each sentence below, indicate whether the sentiment is negative (0) or positive (1).

Sentence: {sentence}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [self.preamble.format(sentence=e["sentence"]) for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


sst2 = SST2Dataset


class WNLIDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("glue", "wnli")
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            2,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """For each pair of sentences below, indicate whether the second entails (1) or does not entail (0) the first.

Sentence 1: Steve follows Fred's example in everything. He influences him hugely.
Sentence 2: Steve influences him hugely.
Answer: 0

Sentence 1: The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood.
Sentence 2: The police were trying to stop the drug trade in the neighborhood.
Answer: 1

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    zero_shot_preamble = """For each pair of sentences below, indicate whether the second entails (1) or does not entail (0) the first.

Sentence 1: {sentence_1}
Sentence 2: {sentence_2}
Answer:"""

    def clm_collate_fn(self, batch):
        # No need to use self.add_space here since we add it to the target tokens
        prompts = [
            self.preamble.format(
                sentence_1=e["sentence1"], sentence_2=e["sentence2"])
            for e in batch
        ]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        targets = t.cat([self.label_idx2target_id[e["label"]] for e in batch])
        return prompts, classes, targets

    def sc_collate_fn(self, batch):
        prompts = [e["sentence1"] + " " + e["sentence2"] for e in batch]
        # prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
        # prompts = {k: v[:, -self.max_len :] for k, v in prompts.items()}
        classes = t.tensor([e["label"] for e in batch])
        return prompts, classes, None


wnli = WNLIDataset


class TrivaQADataset(ClassificationDataset):
    """Modified the code here to get the model to predict the text for the right answer rather than just the label."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):

        # Load the original train and validation splits
        dset = load_dataset("lucadiliello/triviaqa")

        self.zero_shot_preamble = """You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words.
Question: {question}"""

        
        few_shot_examples = [
            ["Which actress has won more Best Actress Oscars than anyone else?", "A group of which birds is known as a Muster or a Phalanx?", "Murdered by a bomb in a small fishing boat off County Sligo, in which year did Lord Louis Mountbatten die?", "Brazilian football legend Pele always wore which number", "Which British monarch founded the Order of the Bath?", "How many of William the Conqueror's sons became king of England", 
             "What is the official London residence of the British monarch?", "How old was Jimi Hendrix when he died in 1970?", "The name of which classic film (starring Audrey Hepburn) was also the title of a 1996 hit record for the group 'Deep Blue Something'?", "At which English university was J. R. R. Tolkein a professor when he wrote The Lord Of The Rings?"], 
            ["katharine hepburn", "storks", "1979", "10", "george i", "two", "buckingham palace", "27", "breakfast at tiffany's", "the university of oxford"]
        ]
        
        # Create these as messages 
        messages = [{"role": "system", "content": "You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words."}]
        for question, answer in zip(*few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"Question: {question}"
            })
            messages.append({
                "role": "assistant",
                "content": f"{answer}."
            })
        
        self.few_shot_preamble = messages

        self.preamble = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        # Proceed with the rest of the initialization
        self.dset = dset

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        super().__init__(
            dset=self.dset,
            tokenizer=tokenizer,
            n_labels=5,
            preamble=prompt,
            add_space=add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
            seed=seed,
        )

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            prompts.append(self.preamble.format(question=e["question"]))
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answers"])
            normalized_answers.append(e["answers"])

        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answers"])
            normalized_answers.append(e["answers"])

        return prompts, correct_answers, normalized_answers


trivia_qa = TrivaQADataset


class NaturalQuestionsDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("google-research-datasets/nq_open")
        
        # Restric the training set size to be ~64k examples.
        train_set = dset["train"].shuffle(42).select(range(64000))
        validation_set = dset["validation"]
        
        dset["train"] = train_set
        dset["validation"] = validation_set

        zero_shot_preamble = """You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words.

Question: {question}?"""

        self.zero_shot_preamble = zero_shot_preamble

        few_shot_examples = [
            ["When was the last amendment made to the us constitution?", "What is the second biggest state in united states?", "Which delta is the largest delta in the world?", "How many episodes in american horror story season 1?", "Who starred in breakfast at tiffany's 1961?", "who killed king joffrey on game of thrones", "When is the olympics coming to los angeles?", "Who said the sun revolved around the earth?", "What street did the brady bunch live on?", "Who took over the english throne after the glorious revolution?"], 
            ["May 5, 1992", "Texas", "The Ganges-Brahmaputra Delta", "12", "Audrey Hepburn", "Ser Dontos Hollard", "2028", "Ptolemy", "4222 Clinton Way", "William III"]
        ]
        
        # Create these as messages 
        messages = [{"role": "system", "content": "You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words."}]
        for question, answer in zip(*few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"Question: {question}"
            })
            messages.append({
                "role": "assistant",
                "content": f"{answer}."
            })
        
        self.few_shot_preamble = messages

        self.preamble = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        # Proceed with the rest of the initialization
        self.dset = dset

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        
        super().__init__(
            dset,
            tokenizer,
            1,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )


    def _format_prompts(self, batch):
        prompts = [self.preamble.format(question=e["question"].strip().capitalize()) for e in batch]
        return prompts

    def _normalize_answer(self, answer):
        # Convert to lowercase
        answer = answer.lower()
        # Remove punctuation and brackets
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["answer"] for e in batch]  # Return all answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["answer"] for e in batch]  # Return all answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers


natural_questions = NaturalQuestionsDataset


class SQuADDataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        max_context_len: int = 850,  # New parameter for maximum context length
        seed: int = 42,
    ):
        dset = load_dataset("rajpurkar/squad")
        self.zero_shot_preamble = """You are tasked with answering questions based on provided context text with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words.

Context: {context}

Question: {question}"""

        few_shot_examples = [
            ["Chopin's life was covered in a BBC TV documentary Chopin – The Women Behind The Music (2010), and in a 2010 documentary realised by Angelo Bozzolini and Roberto Prosseda for Italian television.", "The Boudhanath, (also written Bouddhanath, Bodhnath, Baudhanath or the Khāsa Chaitya), is one of the holiest Buddhist sites in Nepal, along with Swayambhu. It is a very popular tourist site. Boudhanath is known as Khāsti by Newars and as Bauddha or Bodhnāth by speakers of Nepali. Located about 11 km (7 mi) from the center and northeastern outskirts of Kathmandu, the stupa's massive mandala makes it one of the largest spherical stupas in Nepal. Boudhanath became a UNESCO World Heritage Site in 1979.", "Most capacitors have numbers printed on their bodies to indicate their electrical characteristics. Larger capacitors like electrolytics usually display the actual capacitance together with the unit (for example, 220 μF). Smaller capacitors like ceramics, however, use a shorthand consisting of three numeric digits and a letter, where the digits indicate the capacitance in pF (calculated as XY × 10Z for digits XYZ) and the letter indicates the tolerance (J, K or M for ±5%, ±10% and ±20% respectively).", "Himachal has a rich heritage of handicrafts. These include woolen and pashmina shawls, carpets, silver and metal ware, embroidered chappals, grass shoes, Kangra and Gompa style paintings, wood work, horse-hair bangles, wooden and metal utensils and various other house hold items. These aesthetic and tasteful handicrafts declined under competition from machine made goods and also because of lack of marketing facilities. But now the demand for handicrafts has increased within and outside the country."],
            ["What was the name of the documentary released by the BBC?", "When did UNESCO make Boudhanath a World Heritage Site?", "What part of the electrical characteristics of smaller capacitors do the digits of the abbreviated notation represent ?", "What declined under competition?"],
            ["Chopin – The Women Behind The Music", "1979", "the digits indicate the capacitance", "aesthetic and tasteful handicrafts"]
        ]
        
        messages = [{"role": "system", "content": "You are tasked with answering questions based on provided context text with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words."}]

        for context, question, answer in zip(*few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}"
            })
            messages.append({
                "role": "assistant",
                "content": f"{answer}."
            })
            
        self.few_shot_preamble = messages
        
        self.preamble = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        self.max_context_len = max_context_len  # Store the max context length
        dset = self._filter_dataset(dset)  # Filter the dataset

        dset["validation"] = dset["validation"].shuffle(seed=seed)

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            1,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )



    def _filter_dataset(self, dset):
        """Filter out examples with excessively long contexts."""

        def filter_fn(example):
            # Filter out examples with context longer than max_context_len
            return len(example["context"]) <= self.max_context_len

        dset = dset.filter(filter_fn)
        return dset

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            prompts.append(
                self.preamble.format(
                    context=e["context"].strip(), question=e["question"].strip()
                )
            )
        return prompts

    def _normalize_answer(self, answer):
        # Convert to lowercase
        answer = answer.lower()
        # Remove punctuation and brackets
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["answers"]["text"]
                           for e in batch]  # Return all answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["answers"]["text"]
                           for e in batch]  # Return all answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers


squad = SQuADDataset


class CoQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        max_context_len: int = 1610,
        seed: int = 42,
    ):
        # Load both train and validation splits
        train_dset = load_dataset("stanfordnlp/coqa", split="train")
        val_dset = load_dataset("stanfordnlp/coqa", split="validation")

        self.max_context_len = max_context_len

        # Process and filter both splits
        train_dset = self._process_and_filter_dataset(train_dset)
        val_dset = self._process_and_filter_dataset(val_dset)

        # Combine into a DatasetDict
        dset = datasets.DatasetDict(
            {"train": train_dset, "validation": val_dset})

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            1,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    few_shot_preamble = """Answer the question below based on the given context, providing a short and concise answer.

Context: {context}
Question: {input_text}"""

    zero_shot_preamble = """Answer the question below based on the given context, providing a short and concise answer.

Context: {context}
Question: {input_text}"""

    def _process_and_filter_dataset(self, dset):
        """Process and filter the dataset by splitting questions and applying filters."""

        new_examples = []

        for example in dset:
            if "questions" in example and "answers" in example:
                for i, question in enumerate(example["questions"]):
                    # Create a new example for each question-answer pair
                    if len(example["story"]) <= self.max_context_len:
                        new_example = {
                            "context": example[
                                "story"
                            ],  # Use the same context for each pair
                            "input_text": question,  # The specific question
                            "output": [example["answers"]["input_text"][
                                i
                            ]],  # Corresponding answer
                        }
                        new_examples.append(new_example)

        # Shuffle the new examples list
        random.shuffle(new_examples)

        # Create a new Dataset object
        dset = datasets.Dataset.from_dict(
            {
                "context": [ex["context"] for ex in new_examples],
                "input_text": [ex["input_text"] for ex in new_examples],
                "output": [ex["output"] for ex in new_examples],
            }
        )

        return dset

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            prompts.append(
                self.preamble.format(
                    context=e["context"].strip(), input_text=e["input_text"].strip()
                )
            )
        return prompts

    def _normalize_answer(self, answer):
        # Convert to lowercase
        answer = answer.lower()
        # Remove punctuation and brackets
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["output"]
                           for e in batch]  # Get 'output' as answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["output"]
                           for e in batch]  # Get 'output' as answers
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers]
            for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers


coqa = CoQADataset

class RedTrivaQADataset(ClassificationDataset):
    """Modified the code here to get the model to predict the text for the right answer rather than just the label."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):

        # Load the original train and validation splits
        dset = load_dataset("lucadiliello/triviaqa")

        # Note the sizes of the original train and validation sets
        original_train_size = len(dset["train"])
        original_val_size = len(dset["validation"])

        # Combine train and validation splits
        combined_dataset = datasets.concatenate_datasets(
            [dset["train"], dset["validation"]])

        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=seed)

        # Split back into train and validation sets of the same sizes as before
        new_train_set = combined_dataset.select(range(0, original_train_size))
        new_validation_set = combined_dataset.select(
            range(original_train_size, original_train_size + original_val_size))

        # Update self.dset with the new splits
        dset["train"] = new_train_set
        dset["validation"] = new_validation_set

        # Proceed with the rest of the initialization
        self.dset = dset

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        super().__init__(
            dset=self.dset,
            tokenizer=tokenizer,
            n_labels=5,
            preamble=prompt,
            add_space=add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
            seed=seed,
        )

    few_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: Who found the remains of the Titanic?
Answer: Bob Ballard

Question: Which country does musician Alfred Brendel come from?
Answer: The Republic of Austria

Question: {question}"""

    zero_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: {question}"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            prompts.append(self.preamble.format(question=e["question"]))
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answer"]["aliases"])
            normalized_answers.append(e["answer"]["normalized_aliases"])

        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answer"]["aliases"])
            normalized_answers.append(e["answer"]["normalized_aliases"])

        return prompts, correct_answers, normalized_answers


red_trivia_qa = RedTrivaQADataset

class RedTrivaQADatasetExt(ClassificationDataset):
    """Modified the code here to get the model to predict the text for the right answer rather than just the label."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):

        # Load the original train and validation splits
        dset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

        # Note the sizes of the original train and validation sets
        original_train_size = len(dset["train"])
        original_val_size = len(dset["validation"])

        # Combine train and validation splits
        combined_dataset = datasets.concatenate_datasets(
            [dset["train"], dset["validation"]])

        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=seed)

        # Split back into train and validation sets of the same sizes as before
        new_train_set = combined_dataset.select(range(0, 100000))
        new_validation_set = combined_dataset.select(
            range(original_train_size, original_train_size + original_val_size))

        # Update self.dset with the new splits
        dset["train"] = new_train_set
        dset["validation"] = new_validation_set

        # Proceed with the rest of the initialization
        self.dset = dset

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        super().__init__(
            dset=self.dset,
            tokenizer=tokenizer,
            n_labels=5,
            preamble=prompt,
            add_space=add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
            seed=seed,
        )

    few_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: Who found the remains of the Titanic?
Answer: Bob Ballard

Question: Which country does musician Alfred Brendel come from?
Answer: The Republic of Austria

Question: {question}"""

    zero_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: {question}"""

    def _format_prompts(self, batch):
        prompts = []
        for e in batch:
            prompts.append(self.preamble.format(question=e["question"]))
        return prompts

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answer"]["aliases"])
            normalized_answers.append(e["answer"]["normalized_aliases"])

        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        normalized_answers = []
        correct_answers = []
        for e in batch:
            correct_answers.append(e["answer"]["aliases"])
            normalized_answers.append(e["answer"]["normalized_aliases"])

        return prompts, correct_answers, normalized_answers


red_trivia_qa_ext = RedTrivaQADatasetExt

class GSM8KDataset(ClassificationDataset):
    """
    A dataset wrapper for openai/gsm8k, 'main' configuration (train and test splits).
    
    Since GSM8K is a free-form QA (math word-problem) dataset, we treat it similarly
    to other extractive or generative QA datasets (e.g. SQuAD or TriviaQA).
    
    The 'answer' column in GSM8K often includes a chain-of-thought (explanation) plus
    the final numerical answer. Here, we simply provide the question in the prompt and
    return the entire 'answer' field as the ground-truth solution. You can adapt this
    behavior as you see fit (for example, by post-processing to extract only the final
    answer).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):
        # Load the dataset from the Hugging Face Hub
        # 'main' is the configuration for the standard GSM8K training and test sets.
        dset = load_dataset("openai/gsm8k", "main")

        # We define a default prompt. You can choose to do a zero-shot or few-shot approach.
        # For demonstration, we mimic the style of short QA prompts (like SQuAD/TriviaQA).
        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        # We call the parent constructor. n_labels=1 because we have no distinct classification
        # labels—only a single textual answer. numerical=False means we won't treat it as "0, 1, 2, ...".
        super().__init__(
            dset=dset,
            tokenizer=tokenizer,
            n_labels=1,
            preamble=prompt,
            add_space=add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
            seed=seed,
        )

    few_shot_preamble = """Answer the question below. Provide a clear and concise final result.

Example 1:
Question: There are 3 apples and 4 oranges. How many fruits in total?
Answer: 7

Example 2:
Question: If Maria has 5 pencils and buys 3 more boxes with 2 pencils each, how many pencils does she have now?
Answer: 11

Question: {question}
Answer:"""

    zero_shot_preamble = """Solve the following math problem step by step. When done, print only “####” and then give your final answer only. Your output should be of the form [reasoning] #### [final_answer].

Problem: {question}
Answer:"""

    def _format_prompts(self, batch):
        """
        Build the prompt strings for each example in the batch using the
        (few-shot or zero-shot) preamble plus the question.
        """
        prompts = []
        for e in batch:
            prompts.append(
                self.preamble.format(question=e["question"])
            )
        return prompts

    def _normalize_answer(self, answer):
        """
        Optional normalization routine if you want to canonicalize the final
        numeric answer. For GSM8K, the 'answer' field often has chain of thought
        included as plain text. If you want just the final numeric answer, you
        must parse it out here. By default, we leave it as-is.
        """
        # For example, if you want to attempt a naive parse of the last integer:
        # import re
        # m = re.search(r'(\d+)\s*$', answer.strip())
        # if m:
        #     return m.group(1)
        # return answer.strip()
        return answer.strip()

    def clm_collate_fn(self, batch):
        """
        Causal language modeling style collation. The returned `targets` can be
        the correct step-by-step solution or just the final numeric answer,
        depending on your usage.
        
        Returns:
            prompts: List of textual prompts (strings).
            correct_answers: List of full raw answers (strings) from the dataset.
            normalized_answers: List of normalized answers (strings) if used.
        """
        prompts = self._format_prompts(batch)
        
        # GSM8K has 'answer' as a string containing the chain-of-thought plus final answer.
        # We'll provide the entire thing as ground truth. Modify if you want only the final number.
        correct_answers = [[e["answer"]] for e in batch]
        normalized_answers = [
            [answer]
            for answer in correct_answers
        ]

        return prompts, correct_answers, correct_answers

    def sc_collate_fn(self, batch):
        """
        Sequence classification style collation—here, somewhat degenerate since
        GSM8K is not a classification dataset. We'll still return the question as
        prompts, and the correct_answers in the same structure as clm_collate_fn.
        """
        prompts = self._format_prompts(batch)
        correct_answers = [[e["answer"]] for e in batch]
        normalized_answers = [
            [answer]
            for answer in correct_answers
        ]

        # We return `None` in place of numeric class labels since there's no
        # discrete classification label here (like 0/1/2).
        return prompts, correct_answers, correct_answers


# Finally, expose gsm8k in the local namespace if desired
gsm8k = GSM8KDataset

from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

### TODO Currently just laoding trivia!!!!!!!!!!!!!!!!!!!!!!!!!!
class AmortizedClassificationDataset:
    few_shot_preamble = """Question: {question}"""
    zero_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: {question}"""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = False,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.add_space = add_space
        self.few_shot = few_shot
        self.max_len = max_len
        self.seed = seed

        # Select prompt style
        self.preamble = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        # Load and combine training datasets
        tqa = load_dataset("lucadiliello/triviaqa")
        nq  = load_dataset("google-research-datasets/nq_open")
        
        # Map TriviaQA to retain only "question" and "answer"
        def map_tqa(example):
            return {"question": example["question"], "answer": example["answers"]}
        
        train_tqa = tqa["train"].map(
            map_tqa,
            remove_columns=[col for col in tqa["train"].column_names if col not in ["question", "answers"]]
        )
        
        # For TriviaQA validation split, apply the same transformation.
        validation_tqa = tqa["validation"].map(
            map_tqa,
            remove_columns=[col for col in tqa["validation"].column_names if col not in ["question", "answers"]]
        )
        
        # For Natural Questions, we assume the fields are already consistent.
        train_nq = nq["train"].shuffle(seed=seed).select(range(64000))
        
        print(f"TriviaQA raw train size: {len(train_tqa)}")
        print(f"NQ raw train size: {len(train_nq)}")
        
        # Concatenate training splits and shuffle
        combined_dataset = concatenate_datasets([train_tqa]).shuffle(seed=seed)
        print(f"Combined dataset size: {len(combined_dataset)}")
        
        # Split off 3000 examples for validation
        combined_size = len(combined_dataset)
        train_size = combined_size - 3000
        self.train_dataset = combined_dataset.select(range(train_size))
        self.validation_dataset = combined_dataset.select(range(train_size, combined_size))

        # Prepare test splits (apply mapping for TriviaQA validation split too)
        self.trivia_test = validation_tqa.shuffle(seed=seed).select(range(3000))
        # self.nq_test     = nq["validation"].shuffle(seed=seed).select(range(3000))

    def _format_prompts(self, batch):
        # Apply the preamble formatting for each question
        return [self.preamble.format(question=e["question"]) for e in batch]

    def _normalize_answer(self, answer):
        # Convert to lowercase
        answer = answer.lower()
        # Remove punctuation and brackets
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        # Remove extra whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        # Extract answers: TriviaQA uses 'answers', NQ uses 'answer'
        answers = []
        for e in batch:
            answers.append(e["answer"])
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answer_group]
            for answer_group in answers
        ]
        return prompts, answers, normalized_answers

    def loader(self, batch_size: int = 8, shuffle: bool = True):
        """
        Returns:
          - 'train': combined train loader
          - 'validation': validation loader (2000 examples from train)
          - 'trivia_test': TriviaQA test loader
          - 'nq_test': NQ test loader
        """
        return {
            "train": DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.clm_collate_fn,
            ),
            "validation": DataLoader(
                self.validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.clm_collate_fn,
            ),
            "trivia_test": DataLoader(
                self.trivia_test,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.clm_collate_fn,
            ),
            # "nq_test": DataLoader(
            #     self.nq_test,
            #     batch_size=batch_size,
            #     shuffle=False,
            #     collate_fn=self.clm_collate_fn,
            # ),
        }


class AmbigQADataset(ClassificationDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        add_space: bool = True,
        few_shot: bool = False,
        max_len: int = 4096,
    ):
        dset = load_dataset("sewon/ambig_qa")

        self.zero_shot_preamble = """You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words.
Question: {question}"""

        
        few_shot_examples = [
            ["Who said the unexamined life isn't worth living?", "Who wrote the song i kissed a girl?", "Where was the winter olympics held in the united states?", "Who were the major prophets in the bible?", "When was the first airplane used in war?", "What size is a california king size mattress?", "Who played young tom riddle in harry potter?", "The colorado plateau covers northern new mexico and what other state?", "When does the movie the last jedi come out?" "What was the original color of the golden gate bridge?"], 
            ["Socrates", "Cathy Dennis", "Salt Lake City", "Isaiah", "1911", "72 in × 84 in", "Hero Fiennes-Tiffin", "Arizona", "December 15, 2017", "Orange"]
        ]
        
        # Create these as messages 
        messages = [{"role": "system", "content": "You are tasked with answering questions with simple, single phrase responses. Provide only the exact answer to the question, ending this with the character “.” (a single period). Do not include justifications, reasoning, or any extra words."}]
        for question, answer in zip(*few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"Question: {question}"
            })
            messages.append({
                "role": "assistant",
                "content": f"{answer}."
            })
        
        self.few_shot_preamble = messages

        self.dset = dset

        prompt = self.few_shot_preamble if few_shot else self.zero_shot_preamble
        super().__init__(
            dset,
            tokenizer,
            1,
            prompt,
            add_space,
            numerical=False,
            few_shot=few_shot,
            max_len=max_len,
        )

    def _format_prompts(self, batch):
        prompts = [self.preamble.format(question=e["question"]) for e in batch]
        return prompts

    def _normalize_answer(self, answer):
        answer = answer.lower()
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["nq_answer"] for e in batch]
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers] for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers

    def sc_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["nq_answer"] for e in batch]
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers] for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers


ambig_qa = AmbigQADataset


class AmbigQADatasetPTCalibration:
    few_shot_preamble = """Question: {question}"""
    zero_shot_preamble = """Answer the question below, providing a short and concise answer.

Question: {question}"""

    def __init__(
        self,
        tokenizer,
        add_space: bool = False,
        few_shot: bool = False,
        max_len: int = 4096,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.add_space = add_space
        self.few_shot = few_shot
        self.max_len = max_len
        self.seed = seed
        
        few_shot_examples = [
            ["How many stars does the flag of honduras have?", "Who wrote the new york state constitution of 1777?", "What is the scientific name for a red fox?", "What books of the bible are included in the torah?", "Who wrote the song i kissed a girl?", "How many us sailors died at pearl harbor?", 
             "What size is a california king size mattress?", "Where did the ottoman empire extend to at its peak?"], 
            ["5", "Robert R. Livingston", "Vulpes vulpes", "Deuteronomy", "Cathy Dennis", "2,008 sailors", "72 in × 84 in", "North Africa"]
        ]
        
        # Create these as messages 
        messages = [{"role": "system", "content": "Answer the question below, providing a short and concise answer."}]
        for question, answer in zip(*few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"Question: {question}"
            })
            messages.append({
                "role": "assistant",
                "content": f"{answer}."
            })
        
        self.few_shot_preamble = messages

        self.preamble = self.few_shot_preamble if few_shot else self.zero_shot_preamble

        # Load and filter
        raw = load_dataset("sewon/ambig_qa")
        
        # Might add this in later, for now keep ignore in order to have a larger dataset.
        # raw["train"] = raw["train"].filter(lambda ex: len(ex["nq_answer"]) > 1)
        # raw["validation"] = raw["validation"].filter(lambda ex: len(ex["nq_answer"]) > 1)

        # Split train into train and validation
        split = raw["train"].train_test_split(test_size=0.1, seed=seed)
        self.train_dataset = split["train"]
        self.validation_dataset = split["test"]
        self.test_dataset = raw["validation"]
        
        # Print the lengths of this.
        print(len(self.train_dataset))
        print(len(self.validation_dataset))
        print(len(self.test_dataset))

    def _format_prompts(self, batch):
        return [self.preamble.format(question=e["question"]) for e in batch]

    def _normalize_answer(self, answer):
        answer = answer.lower()
        answer = re.sub(f"[{re.escape(string.punctuation)}]", "", answer)
        answer = re.sub(r"[\[\]\(\)\{\}]", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def clm_collate_fn(self, batch):
        prompts = self._format_prompts(batch)
        correct_answers = [e["nq_answer"] for e in batch]
        normalized_answers = [
            [self._normalize_answer(ans) for ans in answers] for answers in correct_answers
        ]
        return prompts, correct_answers, normalized_answers

    def loader(self, batch_size: int = 8, shuffle: bool = True):
        return {
            "train": DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.clm_collate_fn,
            ),
            "validation": DataLoader(
                self.validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.clm_collate_fn,
            ),
            "test": DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.clm_collate_fn,
            ),
        }