import wandb  # Ensure you've logged in using `wandb login`  # type: ignore
import sys
import os

# Ensure repository root is on sys.path so absolute imports like
# `from src.utils...` work whether this file is executed directly or as a module.
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import utilities via the package path (avoids relative import errors when
# running the script directly).
from src.utils.calibration_evaluation import evaluate_model_and_save_results
from src.utils.utils import set_seed
from src.utils.text_utils import exclude_instruction
from src.utils import dsets  # Your custom dataset module
import random
from argparse import ArgumentParser
import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)

# Import the calibration model/config from the installed llmtuner package
# (from the adaptive-temperature-scaling submodule)
from llmtuner.model.calibration.modeling import CalibrationModelForCausalLM
from llmtuner.model.calibration.config import CalibrationConfig

# -------------------------------
# Argument Parsing
# -------------------------------
def parse_args():
    parser = ArgumentParser(
        description="LLaMA with GQA, RoPE, and Temperature Head Training Script")
    parser.add_argument("--output_dir", type=str,
                        default="results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument(
        "--wandb_project", type=str, default="calibration_of_pre_trained_model", help="WandB project name"
    )
    parser.add_argument("--dset", type=str,
                        default="trivia_qa", help="Dataset name")
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
    parser.add_argument("--num_bins", type=int, default=15,
                        help="Number of bins for classification (if applicable)")
    parser.add_argument("--max_new_tokens", type=int,
                        default=10, help="Maximum new tokens for generation")
    parser.add_argument("--train_temp", action="store_false",
                        help="Whether to train the temperature head")
    parser.add_argument("--wandb", action="store_false",
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--load_original_model", action="store_true",
                        help="Whether to load the original model without modifications")
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Whether to use Few-Shot Learning for MAP calculations.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--temp_lr", type=float, default=1e-5,
                        help="Learning rate for temperature head")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Regularization with adamw")
    parser.add_argument("--temp_epochs", type=int, default=2,
                        help="Number of epochs for temperature optimization")
    parser.add_argument("--temp_head_type", type=str, default="transformer",
                        choices=["platt", "transformer"],
                        help="Choose between 'linear' and 'attention' for the temperature head.")
    parser.add_argument("--loss_weight", type=float,
                        default=0.5, help="Alpha value for adaptive loss")
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["adaptive", "cross_entropy", "ce_with_select_smoothing",
                 "ce_with_logit_selective_smoothing", "ce_with_sentence_logit_smoothing_normalised"],
        default="adaptive",
        help="Choose between 'adaptive' and 'cross_entropy' loss."
    )
    parser.add_argument("--max_seq_len", type=int,
                        default=2048, help="Maximum sequence length")
    parser.add_argument("--multiple_of", type=int, default=256,
                        help="Make SwiGLU hidden layer size multiple of this number")
    parser.add_argument("--ffn_dim_multiplier", type=float,
                        default=None, help="Multiplier for FFN hidden dimension")
    parser.add_argument("--norm_eps", type=float, default=1e-5,
                        help="Epsilon for layer normalization")
    parser.add_argument("--n_kv_heads", type=int, default=8,
                        help="Number of key/value heads")

    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation with different seeds."
    )
    parser.add_argument(
        "--generation_numbers",
        nargs="+",
        type=int,
        default=[10],
        help="Number of generations to sample",
    )
    parser.add_argument(
        "--nli_model_name",
        type=str,
        default="microsoft/deberta-v2-xxlarge-mnli",
        choices=["microsoft/deberta-v2-xxlarge-mnli", "tasksource/ModernBERT-large-nli"],
        help="Name of the NLI model to use for entailment.",
    )
    # Add head type argumetn here
    args = parser.parse_args()

    return args


# -------------------------------
# Optimize Temperature Function
# -------------------------------
def optimize_temperature(args, model, model_id, tokenizer, calib_train_loader, calib_early_stopping_loader, device, support_examples=None):
    # Do not modify anything else here
    temp_optimizer = optim.AdamW(
        model.calibration_head.parameters(), lr=args.temp_lr, weight_decay=args.weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        temp_optimizer,
        num_warmup_steps=int(0.05 * len(calib_train_loader)), # Half an epoch warmup.
        num_training_steps= len(calib_train_loader) * args.temp_epochs,
        num_cycles=0.5,  
        last_epoch=-1,
    )

    loss_save_dir = f"{args.loss_type}_temp_loss" if args.loss_type == "cross_entropy" else f"{args.loss_type}_temp_loss"

    save_directory = os.path.join(
        "/scratch/local/ssd/tomlamb",
        "semantic_calibration",
        "models",
        model_id,
        args.dset,
        f"{loss_save_dir}_head_{args.temp_head_type}",
        f"loss_weight_{args.loss_weight}_temp_lr_{args.temp_lr}_weight_decay{args.weight_decay}"
    )
    os.makedirs(save_directory, exist_ok=True)

    for epoch in range(args.temp_epochs):
        model.train()
        total_loss = 0
        grad_steps = 0
        

        for idx, (prompts, classes, _) in enumerate(tqdm(calib_train_loader, desc=f"Epoch {epoch + 1}")):
            
            temp_optimizer.zero_grad()

            if args.dset in ["trivia_qa", "natural_questions", "squad", "coqa", "red_trivia_qa_ext", "gsm8k", "ambig_qa"]:
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
                                
            tokenized_inputs_with_answers = tokenizer(
                messages, add_special_tokens=True, return_tensors="pt", padding=True).to(device)

            if "falcon" in model_id:
                answer_strings = [
                    " " + c + '.' + tokenizer.eos_token for c in classes
                ]
            else:
                answer_strings = [
                    c + '.' + tokenizer.eos_token for c in classes
                ]

            tokenized_class_lengths = torch.tensor(
                [
                    len(tokenizer(answer, add_special_tokens=False)["input_ids"]) for answer in answer_strings
                ]
            ).to(device) 

            labels = tokenized_inputs_with_answers.input_ids.clone()
            for i, length in enumerate(tokenized_class_lengths):
                if "Qwen" in model_id or "Phi" in model_id:
                    # Some Qwen-based models shift slightly differently
                    labels[i, :-length - 1] = -100
                elif "falcon" in model_id:
                    labels[i, :-length + 1 ] = -100
                else:
                    labels[i, :-length] = -100

            outputs = model(
                input_ids=tokenized_inputs_with_answers.input_ids,
                attention_mask=tokenized_inputs_with_answers.attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss

            if args.wandb:
                wandb.log({"Calibration Training Loss": loss.item()})

            loss.backward()
            # Clip toptimizer gradients
            torch.nn.utils.clip_grad_norm_(model.calibration_head.parameters(), 1.0)
            temp_optimizer.step()
            scheduler.step()
            if args.wandb:
                wandb.log({"temp_lr": temp_optimizer.param_groups[0]["lr"]})

            total_loss += loss.item()
            grad_steps += 1

        avg_loss = total_loss / grad_steps

        if args.wandb:
            wandb.log({"Average Training Loss": avg_loss})
            
        model.save_pretrained(save_directory)

    print("Loading the best calibration head.")
    model.load_calibration_head(save_directory, is_trainable=False)
    print("Training completed. Best calibration head loaded.")
    return model


# -------------------------------
# Main Function
# -------------------------------
def main(args):
    set_seed(args.seed)
    llm_device = "cuda:0"
    nli_device = "cuda:1"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Extract the model ID from the given model name
    model_id = args.model_name.split("/")[-1]

    if args.load_original_model:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16
        ).to(llm_device)
        base_model_name = args.model_name
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            f"results_repeats/{model_id}/{args.dset}/ft/ft_acc_model/", torch_dtype=torch.bfloat16
        ).to(llm_device)
        base_model_name = f"results/{model_id}/{args.dset}/ft/ft_acc_model/"
        
    print("Num layers:", base_model.config.num_hidden_layers)
        
    loss_type = args.loss_type if args.loss_type != "adaptive" else "selective_smoothing"
    if args.loss_type == "cross_entropy":
        loss_type = "xent"
        
    # Set the layer index.
    layer_idx = base_model.config.num_hidden_layers
            
    # calibration_config = CalibrationConfig(
    #     base_model_name_or_path=base_model_name,
    #     task_type="CAUSAL_LM",
    #     calibration_type="transformer",
    #     intermediate_size=14336,
    #     loss_type=loss_type,
    #     label_smoothing_type="uniform",
    #     hidden_act='silu',
    #     smooth_loss_weight=args.loss_weight,
    #     num_key_value_heads=8,
    #     feature_key= "hidden_states",
    #     attention_dropout= 0.0,
    #     layer_idx = layer_idx,
    #     max_position_embeddings = 4096,
    #     freeze_base_model=True,
    #     num_attention_heads=32,
    #     init_temperature= 1.0,
    #     in_features=base_model.config.hidden_size,
    #     inference_mode= True
    # )
    
    # Print number of heads in the model
    print(f"Number of attention heads in the model: {base_model.config.num_attention_heads}")
    
    # Transformer head type
    if args.temp_head_type == "transformer":
        calibration_config = CalibrationConfig(
            base_model_name_or_path=base_model_name,
            task_type="CAUSAL_LM",
            calibration_type="transformer",
            intermediate_size=11008,    
            loss_type=loss_type,
            label_smoothing_type="uniform",
            hidden_act='silu',
            smooth_loss_weight=args.loss_weight,
            num_key_value_heads=32,
            feature_key= "hidden_states",
            attention_dropout= 0.0,
            layer_idx = layer_idx,
            max_position_embeddings = 4096,
            freeze_base_model=True,
            num_attention_heads=32,
            init_temperature= 1.0,
            in_features=base_model.config.hidden_size,
            inference_mode= True
        )
    
    # Diagonal Platt scaling head type
    elif args.temp_head_type == "platt":
            calibration_config = CalibrationConfig(
            base_model_name_or_path=base_model_name,
            task_type="CAUSAL_LM",
            calibration_type="platt_scaling_elementwise",
            loss_type=loss_type,
            label_smoothing_type="uniform",
            feature_key="logits", 
            smooth_loss_weight=args.loss_weight,
            freeze_base_model=True,
            init_temperature= 1.0,
            inference_mode= True,
            in_features=base_model.config.vocab_size,
        )
        

    model = CalibrationModelForCausalLM(
        base_model, calibration_config).to(llm_device)
    
    # Print the number of parameters in the calibration head
    print(f"Number of parameters in the calibration head: {sum(p.numel() for p in model.calibration_head.parameters())}")

    model.train()
    model.set_overwrite_logits(set_to=True)
    model.base_model.eval()

    entailment_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v2-xxlarge-mnli"
    ).half().to(nli_device)
    entailment_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v2-xxlarge-mnli"
    )

    for param in entailment_model.parameters():
        param.requires_grad = False

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

        calib_training = torch.utils.data.ConcatDataset(
            [training_early_stopping.dataset, calib_training.dataset, calib_early_stopping.dataset, training_train_loader.dataset, training_validation.dataset]
        )
        
        calib_training = torch.utils.data.DataLoader(
            calib_training,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=final_test.collate_fn,
        )
        
    else:
        support_examples = None
    
    print("Number of Calibration training examples", len(calib_training.dataset))

    if args.train_temp:
        if args.wandb:
            tags = ["temp_head_optimisation", f"datset_{args.dset}", f"loss_fn_{args.loss_type}"]
            wandb.init(
                project=args.wandb_project, config=args, tags=tags
            )

        model = optimize_temperature(
            args, model, model_id, tokenizer, calib_training, calib_early_stopping, llm_device, support_examples
        )
    else:
        print("Skipping temperature head training...")

    model.eval()
    model.set_overwrite_logits(True)

    print("Evaluating on validation and test sets...")
        
    loss_save_dir = f"{args.loss_type}" if args.loss_type == "cross_entropy" else f"{args.loss_type}"

    save_path = os.path.join(
        args.output_dir + "_test_bugged" if len(args.generation_numbers) == 1 else args.output_dir + "_few_shot_test",
        model_id,
        args.dset,
        f"{loss_save_dir}_head_{args.temp_head_type}",
        f"loss_weight_{args.loss_weight}_temp_lr_{args.temp_lr}_weight_decay_{args.weight_decay}",
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
        temperature=None,
        entailment_model=entailment_model,
        entailment_tokenizer=entailment_tokenizer,
        llm_device=llm_device,
        nli_device=nli_device,
        support_examples=support_examples,
        temperature_head = True
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