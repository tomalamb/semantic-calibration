import numpy as np
import torch as t
import random
import torch
from .evaluation_metrics import evaluate_answer
import json
import os
from typing import Any

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    
def save_json(filename: str, obj: Any, dirpath: str):
    path = os.path.join(dirpath, f"{filename}.json")
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)

def accumulate_metrics(aggregated_dict, new_results):
    """
    Recursively accumulate values into lists.
    If aggregated_dict doesn't have a key, create it.
    If value is a number, append it to a list.
    If value is a dictionary, recurse.
    """
    for k, v in new_results.items():
        if isinstance(v, dict):
            if k not in aggregated_dict:
                aggregated_dict[k] = {}
            accumulate_metrics(aggregated_dict[k], v)
        else:
            # v should be numeric here
            if k not in aggregated_dict:
                aggregated_dict[k] = []
            aggregated_dict[k].append(v)
            
            
def compute_is_correct_batched(batch, device, args, tokenizer, model, support_examples = None):
    prompts = []
    ground_truths = []
    # Build prompts and collect ground truths based on dataset type.
    if args.dset in ["trivia_qa"]:
        for question, answers in zip(batch["question"], batch["answers"]):
            prompt = (
                "Answer the question below, providing a short and concise answer.\n\n"
                f"Question: {question}"
            )
            prompts.append(prompt)
            ground_truths.append(answers)
    elif args.dset in ["natural_questions"]:
        for question, answers in zip(batch["question"], batch["answer"]):
            prompt = (
                "Answer the question below, providing a short and concise answer.\n\n"
                f"Question: {question}"
            )
            prompts.append(prompt)
            ground_truths.append(answers)
    elif args.dset in ["coqa"]:
        for context, input_text, answers in zip(batch["context"], batch["input_text"], batch["output"]):
            prompt = (
                "Answer the question below based on the given context, providing a short and concise answer.\n\n"
                f"Context: {context}\nQuestion: {input_text}"
            )
            prompts.append(prompt)
            ground_truths.append(answers)
    elif args.dset in ["squad"]:
        for context, input_text, answer in zip(batch["context"], batch["question"], batch["answers"]):
            prompt = (
                "Answer the question below based on the given context, providing a short and concise answer.\n\n"
                f"Context: {context}\nQuestion: {input_text}"
            )

            prompts.append(prompt)
            ground_truths.append(answer["text"][0])
    else:
        for question, answers in zip(batch.get("question", []), batch["answers"]):
            prompt = (
                "Answer the question below, providing a short and concise answer.\n\n"
                f"Question: {question}"
            )
            prompts.append(prompt)
            ground_truths.append(answers)

    messages = []
    for prompt in prompts:
        message = [{"role": "user", "content": "Question: " + prompt.split("Question: ")[-1].strip() if args.few_shot else prompt}]
                
        messages.append(
            tokenizer.apply_chat_template(
                message if support_examples is None else support_examples + message,
                tokenize=False,
                add_generation_prompt=True
            ))
        
    # Tokenize the batch of prompts.
    input_ids = tokenizer(
        messages,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True
    ).to(device)

    # Generate model responses in batch.
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=5,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    input_len = input_ids.input_ids.shape[1]
    gen_seqs = output_ids.sequences[:, input_len:]
    generated_texts = tokenizer.batch_decode(
        gen_seqs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Evaluate correctness for each example.
    is_corrects = []
    for gen_text, gt in zip(generated_texts, ground_truths):
        normalized_response = clean_and_preprocess_text(gen_text)
        if isinstance(gt, list):
            normalized_answers = [clean_and_preprocess_text(ans) for ans in gt]
            correct = evaluate_answer(normalized_response, normalized_answers, multiple_references=True)
        else:
            correct = evaluate_answer(normalized_response, gt)
        is_corrects.append(int(correct))
    batch["is_correct"] = is_corrects
    return batch    
