import os
import sys
import torch as t
import numpy as np
import wandb
from tqdm import tqdm

# Ensure repo root on path so absolute imports succeed when the script is
# executed directly (python src/scalar_temp_training.py) or as a module.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils import dsets
from src.utils.calibration_losses import (
    adaptive_loss,
    ce_with_select_smoothing_loss,
    ce_with_logit_selective_smoothing_loss,
)
from src.utils.calibration_evaluation import evaluate_model_and_save_results
from src.utils.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
) 
try:
    from src.utils.text_utils import exclude_instruction
except ImportError:
    from src.utils.text_utils import exclude_instruction
from argparse import ArgumentParser
import random

# ----------------------------------------------------------------------------------------------
# Argument Parsing
# ----------------------------------------------------------------------------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--wandb_project", type=str, default="calibration_of_pre_trained_model"
    )
    parser.add_argument("--dset", type=str, default="trivia_qa")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        choices=["meta-llama/Llama-3.1-8B-Instruct",
                 "mistralai/Ministral-8B-Instruct-2410",
                 "Qwen/Qwen2.5-7B-Instruct",
                 "Qwen/Qwen2.5-1.5B-Instruct",
                 "Qwen/Qwen2.5-3B-Instruct",
                 "tiiuae/falcon-7b-instruct",
                 "microsoft/Phi-4-mini-instruct"
        ]            
    )
    parser.add_argument("--num_bins", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--train_temp", action="store_false")
    parser.add_argument(
        "--wandb", action="store_false", help="Whether to use wandb."
    )
    parser.add_argument("--load_original_model", action="store_true")
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use SFT for MAP calculations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp_lr", type=float, default=1e-4)
    parser.add_argument("--temp_epochs", type=int, default=16)
    parser.add_argument("--initial_opt_temp", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=[
            "adaptive",
            "cross_entropy",
            # "ce_with_select_smoothing",
            "ce_with_logit_selective_smoothing",
        ],
        default="adaptive",
        help="Choose between 'adaptive' and 'cross_entropy' loss.",
    )
    parser.add_argument(
        "--loss_weight",
        type=float,
        default=0.5,
        help="Alpha value for loss",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation with different seeds.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Regularization with adamw"
    )
    parser.add_argument(
        "--generation_numbers",
        nargs="+",
        type=list,
        default=[10],
        help="Number of generations to sample",
    )
    # Add argument for NLI model name.
    parser.add_argument(
        "--nli_model_name",
        type=str,
        default="microsoft/deberta-v2-xxlarge-mnli",
        choices=["microsoft/deberta-v2-xxlarge-mnli", "tasksource/ModernBERT-large-nli"],
        help="Name of the NLI model to use for entailment.",
    )
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------------------------
# Temperature Optimization Function
# ----------------------------------------------------------------------------------------------
def optimize_temperature(
    args, model, model_id, tokenizer, calib_train_loader, device, support_examples = None
):
    """
    Optimizes temperature in log-space to ensure T remains positive.
    """
    
    # Initialize log_temperature to the log of the initial temperature
    initial_temperature = args.initial_opt_temp
    log_temperature = t.tensor(
        np.log(initial_temperature),
        requires_grad=True,
        device=device
    )

    temp_optimizer = t.optim.AdamW(
        [log_temperature], lr=args.temp_lr, weight_decay=args.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        temp_optimizer,
        num_warmup_steps=int(0.05 * len(calib_train_loader)),
        num_training_steps= len(calib_train_loader) * args.temp_epochs,
        num_cycles=0.5,  
        last_epoch=-1,
    )

    # Freeze the model's weights
    for param in model.parameters():
        param.requires_grad = False

    for epoch in range(args.temp_epochs):
        total_loss = 0

        # ------------------------
        # TRAIN STEP on val_train_loader (80%)
        # ------------------------
        for prompts, classes, _ in tqdm(calib_train_loader, desc=f"Epoch {epoch + 1}"):
            temp_optimizer.zero_grad()

            # Choose random class for datasets with multiple answers
            if args.dset in [
                "trivia_qa",
                "natural_questions",
                "squad",
                "coqa",
                "red_trivia_qa_ext",
                "gsm8k",
                "ambig_qa"
            ]:
                classes = [random.choice(c) for c in classes]

            messages = []
            for prompt, output in zip(prompts, classes):
                if args.few_shot:
                    prompt = exclude_instruction(prompt, args.dset)
                

                message = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output + '.'},
                ]
                
                messages.append(
                    tokenizer.apply_chat_template(
                        message if support_examples is None else support_examples + message, tokenize=False, add_generation_prompt=False
                    )
                )
            
            print(messages[0])
                                                                    
            tokenized_inputs_with_answers = tokenizer(
                messages, add_special_tokens=False, return_tensors="pt", padding=True
            ).to(device)
                                                
            if "falcon" in model_id:
                answer_strings = [
                    " " + c + '.' + tokenizer.eos_token for c in classes
                ]
            else:
                answer_strings = [
                    c + '.' + tokenizer.eos_token for c in classes
                ]

            tokenized_class_lengths = t.tensor(
                [
                    len(tokenizer(answer, add_special_tokens=False)["input_ids"]) for answer in answer_strings
                ]
            ).to(device)    
                    
            labels = tokenized_inputs_with_answers.input_ids.clone()
            print(labels[0])
            for i, length in enumerate(tokenized_class_lengths):
                if "Qwen" in model_id or "Phi" in model_id:
                    # Some Qwen-based models shift slightly differently
                    labels[i, :-length - 1] = -100
                elif "falcon" in model_id:
                    labels[i, :-length + 1 ] = -100
                else:
                    labels[i, :-length] = -100
            
            print(labels[0])
                
            with t.no_grad():
                unscaled_logits = model(
                    input_ids=tokenized_inputs_with_answers.input_ids,
                    attention_mask=tokenized_inputs_with_answers.attention_mask,
                ).logits

            # We exponentiate log_temperature to get T
            T = log_temperature.exp()
            scaled_logits = unscaled_logits / T
            shift_logits = scaled_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            

            # Choose loss function based on the args.loss_type
            if args.loss_type == "adaptive":
                loss = adaptive_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    alpha=args.loss_weight,
                    ignore_index=-100,
                )
            elif args.loss_type == "ce_with_select_smoothing":
                loss = ce_with_select_smoothing_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    alpha=args.loss_weight,
                    ignore_index=-100,
                )
            elif args.loss_type == "ce_with_logit_selective_smoothing":
                loss = ce_with_logit_selective_smoothing_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    alpha=args.loss_weight,
                    ignore_index=-100,
                )
            elif args.loss_type == "cross_entropy":
                loss_fct = t.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
            else:
                raise ValueError(f"Unknown loss type: {args.loss_type}")

            if args.wandb:
                wandb.log({"Calibration Training Loss": loss.item()})

            loss.backward()
            temp_optimizer.step()
            scheduler.step()
            if args.wandb:
                wandb.log({"temp_lr": temp_optimizer.param_groups[0]["lr"]})
                # log the actual temperature
                wandb.log({"Temperaure": log_temperature.exp().item()})

            total_loss += loss.item()
            
        avg_loss = total_loss / len(calib_train_loader)

        if args.wandb:
            wandb.log(
                {
                    "Temperature": T.item(),
                    "Average Train Loss": avg_loss,
                    "Epoch": epoch + 1,
                }
            )

        final_temperature = log_temperature.exp().item()

    print(f"Optimal temperature found: {final_temperature:.4f}")
    return final_temperature


# ----------------------------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------------------------
def main(args):
    llm_device = "cuda:0"
    nli_device = "cuda:1"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_id = args.model_name.split("/")[-1]

    # ----------------------------------------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------------------------------------
    if args.load_original_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=t.bfloat16,
        ).to(llm_device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            f"results_repeats/{model_id}/{args.dset}/ft/ft_acc_model/", torch_dtype=t.bfloat16
        ).to(llm_device)

    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------------------------------------
    # Load entailment model and tokenizer
    # ----------------------------------------------------------------------------------------------
    if "deberta" in args.nli_model_name:
        entailment_model = AutoModelForSequenceClassification.from_pretrained(
            args.nli_model_name
        ).half().to(nli_device)
        entailment_tokenizer = AutoTokenizer.from_pretrained(
            args.nli_model_name
        )

        for param in entailment_model.parameters():
            param.requires_grad = False
    else:
        # We use pipelein for the ModernBERT model from Tasksource
        from transformers import pipeline
        entailment_model = pipeline(
            "text-classification",
            model=args.nli_model_name,
            device=nli_device,
        )
        entailment_tokenizer = None

    dset_class: dsets.ClassificationDataset = getattr(dsets, args.dset)
    dset = dset_class(tokenizer, add_space=False, few_shot = False)
    loaders = dset.loader(
        is_sc=True,
        batch_size=args.batch_size,
    )
    training_train_loader = loaders["training"]
    training_early_stopping = loaders["early_stopping"]
    training_validation = loaders["validation"]
    calib_training = loaders["calib_training"]
    calib_early_stopping = loaders["calib_early_stopping"]
    calib_validation = loaders["calib_validation"]
    final_test = loaders["final_test"]
    
    # Limit the number of examples in the validation and test sets for testing
    
    # ----------------------------------------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------------------------------------
    if args.few_shot:
        support_examples = dset.few_shot_preamble

        # turn the initial system prompt into a user message,
        # then insert an assistant follow-up at index 1
        if support_examples and support_examples[0]["role"] == "system" and "Ministral" in model_id:
            sys_msg = support_examples.pop(0)
            sys_msg["role"] = "user"
            support_examples.insert(0, sys_msg)
            support_examples.insert(1, {
                "role": "assistant",
                "content": (
                    "I will answer questions with simple, single phrase responses, ending my response with the character “.” (a single period). I won't give any explanations or additional information, just the answer phrase. "
                )
            })
            

        calib_training = t.utils.data.ConcatDataset(
            [training_early_stopping.dataset, calib_training.dataset, calib_early_stopping.dataset, training_train_loader.dataset, training_validation.dataset]
        )
        
        calib_training = t.utils.data.DataLoader(
            calib_training,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=final_test.collate_fn,
        )
        
    else:
        support_examples = None
    
    print("Number of Calibration training examples", len(calib_training.dataset))
    print("Number of Calibration validation examples", len(calib_validation.dataset))
    print("Number of Final test examples", len(final_test.dataset))
    
    # ----------------------------------------------------------------------------------------------
    # Temperature Optimization on 80% of the Validation Set
    # ----------------------------------------------------------------------------------------------
    if args.train_temp:
        tags = ["scalar_temp_optimisation", f"datset_{args.dset}", f"loss_fn_{args.loss_type}"]
        if args.loss_type == "adaptive":
            tags.append(f"loss_weight: {args.loss_weight}")
            
        model.eval()
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                config=args,
                tags=tags,
            )

        temperature = optimize_temperature(
            args,
            model,
            model_id,
            tokenizer,
            calib_training,    
            llm_device,
            support_examples
        )
        
        if args.loss_type == "cross_entropy":
            hyperparamer_dir = "initial_temp_{}_temp_lr_{}_weight_decay_{}".format(
                args.initial_opt_temp, args.temp_lr, args.weight_decay
            ) 
        else:
            hyperparamer_dir = "initial_temp_{}_temp_lr_{}_weight_decay_{}_loss_weight_{}".format(
                args.initial_opt_temp, args.temp_lr, args.weight_decay, args.loss_weight
            )
            
        save_path = os.path.join(
            args.output_dir if len(args.generation_numbers) == 1 else args.output_dir + "_few_shot",
            model_id,
            args.dset,
            f"{args.loss_type}_scalar",
            hyperparamer_dir,
            f"temp_{temperature}",
        )
    else:
        temperature = args.initial_opt_temp
        save_path = os.path.join(
            args.output_dir if len(args.generation_numbers) == 1 else args.output_dir + "_few_shot",
            model_id,
            args.dset,
            f"map_temp_{temperature}"
            if not args.load_original_model
            else f"pre_trained_temp_{temperature}",
            "initial_temp",
        )

    os.makedirs(save_path, exist_ok=True)
    
    del loaders
    
    dset_class: dsets.ClassificationDataset = getattr(dsets, args.dset)
    dset = dset_class(tokenizer, add_space=False, few_shot = False)

    loaders = dset.loader(
        is_sc=True,
        batch_size=max(args.batch_size // 2, 1) if "Phi" not in model_id else args.batch_size,
    )

    calib_validation = loaders["calib_validation"]
    final_test = loaders["final_test"]
    
    evaluate_model_and_save_results(
        save_path= save_path,
        args = args,
        model=model,
        model_id=model_id,
        tokenizer=tokenizer,
        calib_validation=calib_validation,
        final_test=final_test,
        temperature=temperature,
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        llm_device=llm_device,
        nli_device=nli_device,
        support_examples=support_examples,
    )
        

# ----------------------------------------------------------------------------------------------
# Main Script Entry Point
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.generation_numbers == [['0']]:
        args.generation_numbers = [10]
    
    # Depending on whether we load the original model or not, we set the wandb project name.
    if args.load_original_model:
        args.wandb_project = "PT-Calibration"
        args.temp_epochs = 2 
    else:
        args.wandb_project = "SFT-Calibration"
        args.temp_epochs = 8 
    
    print(args.generation_numbers)
    set_seed(args.seed)
    main(args)
