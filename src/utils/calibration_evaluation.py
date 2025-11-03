import torch.nn.functional as F
import torch as t
from tqdm import tqdm
from .evaluation_metrics import (
    compute_ece,
    compute_prr,
    compute_auroc,
    compute_aurac, 
    compute_selective_auroc,
    compute_ace,
    evaluate_answer,
    compute_rejection_accuracy,
    compute_corp_decomposition
)
from .text_utils import (
    clean_and_preprocess_text,
    clean_generated_text,
    exclude_instruction,
    clean_response
)
from .clustering import (
    compute_clusters_and_lls,
    evaluate_one_cluster_method
)
import os
import json
import wandb
from .utils import set_seed, save_json

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
            

def initialize_metrics_dict(generation_numbers):
    def zero_dict():
         return {n_gen: 0 for n_gen in generation_numbers}
    
    def list_dict():
         return {n_gen: [] for n_gen in generation_numbers}
    
    def dict_of_pair_lists():
         return {n_gen: {"correct": [], "confidence": []} for n_gen in generation_numbers}
    
    def dict_of_uncertainty_lists():
         return {n_gen: {"correct": [], "uncertainty": []} for n_gen in generation_numbers}

    measures = ["mgc", "ml_mgc", "h_mgc", "b_mgc", "l_mgc", "median_mgc", "max_mgc", "h_unnorm_mgc", "min_mgc", "gibbs_1_25_mgc", "gibbs_0_75_mgc", "gibbs_0_5_mgc", "cold_0_5_mgc", "cold_0_75_mgc", "cold_1_25_mgc"]

    metrics = {
        # Scalar metrics
        "total_nll_beam_sgc": 0,
        "total_brier_beam_sgc_conf": 0,
        "total_brier_beam_sgc_lnll": 0,
        "total_nlc_beam_sgc": 0,
        "accuracy_beam_sgc": 0,

        # Per-generation zero metrics
        **{f"total_nll_{tag}": zero_dict() for tag in measures},
        **{f"total_brier_{tag}_{score}": zero_dict() for tag in measures for score in ["conf", "lnll"]},
        **{f"total_nlc_{tag}": zero_dict() for tag in measures},
        **{f"accuracy_{tag}": zero_dict() for tag in measures},

        # PRR / ECE / AUROC values per score type
        **{f"{metric}_values_{tag}_{score}": dict_of_pair_lists() for tag in measures for score in ["conf", "lnll"] for metric in ["prr", "ece", "auroc"]},
        **{f"{metric}_values_beam_sgc_{score}": {"correct": [], "confidence": []} 
            for score in ["conf", "lnll"] for metric in ["prr", "ece", "auroc"]},

        # Entropy-based values (response-based & mean‐cluster)
        **{f"{tag}_rb_entropy_values": dict_of_uncertainty_lists() for tag in measures},
        **{f"{tag}_mc_entropy_values": dict_of_uncertainty_lists() for tag in measures},
        **{f"{tag}_rb_entropy_values_SE": dict_of_uncertainty_lists() for tag in measures},
        **{f"{tag}_mc_entropy_values_SE": dict_of_uncertainty_lists() for tag in measures},     

        # Generation result buffers
        "beam_sgc_results": [],
        **{f"{tag}_results": list_dict() for tag in measures}
         }

    return metrics

    
@t.no_grad()
def evaluate_model(
    model,
    model_id,
    tokenizer,
    dataset,
    temperature,
    args,
    entailment_model=None,
    entailment_tokenizer=None,
    device="cuda:0",
    nli_device="cuda:1",
    temperature_head=False,
    generation_numbers = [10],
    top_p = True,
    top_k = True,
    support_examples = None
):
    
    # Prep NLI model
    nli_model_name = args.nli_model_name.lower()
    use_modernbert = "deberta" not in nli_model_name and "modernbert" in nli_model_name
    # (or whatever logic exactly matches your new model's name)
    if use_modernbert:
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="tasksource/ModernBERT-large-nli",
            device=nli_device  # if you need GPU/CPU control
        )
    else:
        pipe = None
    
    # Define the new generation numbers
    metrics = { "normalised": initialize_metrics_dict(generation_numbers)}

    num_examples = 0

    for batch_idx, batch in enumerate(tqdm(dataset)):
        # Remove or adjust this condition as needed
        if len(batch) == 3:
            prompts, classes, normalised_classes = batch
        elif len(batch) == 4:
            prompts, classes, normalised_classes, is_correct = batch
            
        cleaned_gold_lists = [
            [ clean_and_preprocess_text(ans) for ans in ref_list ]
            for ref_list in normalised_classes
        ]
            
        num_examples += len(prompts)

        messages = []
        for input in prompts:
            prompt_minus_instruction = exclude_instruction(input, args.dset).strip()
            message = [
                # Create user-only message
                {"role": "user", "content": prompt_minus_instruction  if args.few_shot else input},
            ]
            
            messages.append(tokenizer.apply_chat_template(
                message if support_examples is None else support_examples + message, tokenize=False,  add_generation_prompt=True))
                                            
        tokenized_inputs = tokenizer(
            messages, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
    
        ###################################################################
        # Beam evaluation.
        ###################################################################
        gen_params = {
            "do_sample": False,
            "num_beams": 4,
            "num_return_sequences": 1,
            "max_new_tokens": args.max_new_tokens,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.eos_token_id,
            "top_p": None,
            "top_k": None,
            "temperature": None,
        }
    
        with t.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                **gen_params
            )

        input_len = tokenized_inputs.input_ids.shape[1]

        beam_sgc_decoded_outputs = tokenizer.batch_decode(
            outputs.sequences[:, input_len:], skip_special_tokens=True
        )
        
        beam_sgc_decoded_outputs = [
            clean_generated_text(text) for text in beam_sgc_decoded_outputs
        ]

        prob_messages = []
        for prompt, output in zip(prompts, beam_sgc_decoded_outputs):
            prompt_minus_instruction = exclude_instruction(prompt, args.dset).strip()
            message = [
                # Create user-only message
                {"role": "user", "content": prompt_minus_instruction if args.few_shot else prompt},
                {"role": "assistant", "content": output}
            ]

            prob_messages.append(tokenizer.apply_chat_template(
                message if support_examples is None else support_examples + message, tokenize=False,  add_generation_prompt=False))
                
        combined_tokenized_outputs = tokenizer(
            prob_messages, add_special_tokens=True, return_tensors="pt", padding=True).to(device)

        if "falcon" in model_id:
            answer_strings = [
                " " + c  + tokenizer.eos_token for c in beam_sgc_decoded_outputs
            ]
        else:
            answer_strings = [
                c  + tokenizer.eos_token for c in beam_sgc_decoded_outputs
            ]

        tokenized_class_lengths = t.tensor(
            [len(tokenizer.tokenize(answer, add_special_tokens=False))
                for answer in answer_strings]
        ).to(device)        

        labels = combined_tokenized_outputs.input_ids.clone()
        for i, length in enumerate(tokenized_class_lengths):
            if "Qwen" in model_id or "Phi" in model_id:
                labels[i, :-length-1] = -100
            elif "falcon" in model_id:
                labels[i, :-length + 1 ] = -100
            else:
                labels[i, :-length] = -100
                                
        with t.no_grad():
            outputs = model(
                input_ids=combined_tokenized_outputs.input_ids,
                attention_mask=combined_tokenized_outputs.attention_mask,
            )

            if not temperature_head:
                scaled_logits = outputs.logits / temperature
            else:
                scaled_logits = outputs.logits

            shift_logits = scaled_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            log_probs = F.log_softmax(shift_logits, dim=-1)

            batch_size, seq_length = shift_labels.size()
            batch_indices = (
                t.arange(batch_size, device=device).unsqueeze(
                    1).expand(-1, seq_length)
            )
            seq_indices = (
                t.arange(seq_length, device=device).unsqueeze(
                    0).expand(batch_size, -1)
            )

            log_probs = log_probs[batch_indices, seq_indices, shift_labels]
            valid_mask = (shift_labels != -100).float()
            masked_log_probs = log_probs * valid_mask
            log_likelihoods = masked_log_probs.sum(dim=-1)

            num_valid_tokens = valid_mask.sum(dim=-1)
            
            # Avoid division by zero
            num_valid_tokens = t.where(
                num_valid_tokens == 0, t.tensor(1.0, device=device), num_valid_tokens)
            
            normalized_log_likelihoods = log_likelihoods / num_valid_tokens
            unnormalized_probabilities = log_likelihoods.exp()
            normalized_probabilities = normalized_log_likelihoods.exp()

        for norm_type in metrics:
            if norm_type == "normalised":
                # Use normalized likelihoods
                probabilities = normalized_probabilities
                current_log_likelihoods = normalized_log_likelihoods
            else:
                # Use unnormalized likelihoods
                probabilities = unnormalized_probabilities
                current_log_likelihoods = log_likelihoods

            # Accumulate total NLL and NLC for beam search
            metrics[norm_type]["total_nll_beam_sgc"] -= current_log_likelihoods.sum().item()
            metrics[norm_type]["total_nlc_beam_sgc"] -= current_log_likelihoods.sum().item()

            # Initialize correctness list for current norm_type
            current_correctness = []
            
            # Fully clean the beam generations her with clean response, and clean and preprocess the text
            beam_sgc_decoded_outputs = [
                clean_and_preprocess_text(clean_response(response, args.dset)) for response in beam_sgc_decoded_outputs
            ]
            
            # Calculate correctness and update metrics
            for i, prompt in enumerate(prompts):
                normalised_final_response = beam_sgc_decoded_outputs[i]
                confidence_beam_sgc = probabilities[i].item()
                final_response_beam_sgc_lnll = current_log_likelihoods[i]
                
                correctness_beam_sgc = evaluate_answer(
                    normalised_final_response,
                    cleaned_gold_lists[i],
                )
    
                current_correctness.append(correctness_beam_sgc)

                # Update accuracy and confidence metrics
                metrics[norm_type]["accuracy_beam_sgc"] += correctness_beam_sgc
                metrics[norm_type]["ece_values_beam_sgc_lnll"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["ece_values_beam_sgc_lnll"]["confidence"].append(
                    final_response_beam_sgc_lnll.exp().item()
                )
                metrics[norm_type]["prr_values_beam_sgc_lnll"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["prr_values_beam_sgc_lnll"]["confidence"].append(
                    final_response_beam_sgc_lnll.exp().item()
                )
                metrics[norm_type]["auroc_values_beam_sgc_lnll"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["auroc_values_beam_sgc_lnll"]["confidence"].append(
                    final_response_beam_sgc_lnll.item()
                )
                metrics[norm_type]["ece_values_beam_sgc_conf"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["ece_values_beam_sgc_conf"]["confidence"].append(
                    confidence_beam_sgc
                )
                metrics[norm_type]["prr_values_beam_sgc_conf"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["prr_values_beam_sgc_conf"]["confidence"].append(
                    confidence_beam_sgc
                )
                metrics[norm_type]["auroc_values_beam_sgc_conf"]["correct"].append(
                    correctness_beam_sgc
                )
                metrics[norm_type]["auroc_values_beam_sgc_conf"]["confidence"].append(
                    confidence_beam_sgc
                )
                metrics[norm_type]["total_brier_beam_sgc_conf"] += (
                    confidence_beam_sgc - correctness_beam_sgc
                ) ** 2
                metrics[norm_type]["total_brier_beam_sgc_lnll"] += (
                    final_response_beam_sgc_lnll.exp().item() - correctness_beam_sgc
                ) ** 2

                # Append results to beam_sgc_results for the current normalization type
                metrics[norm_type]["beam_sgc_results"].append(
                    {
                        "query_text_prompt": messages[i],
                        "generated_text": normalised_final_response,
                        "actual_label": classes[i],
                        "correctness": correctness_beam_sgc,
                        "confidence": confidence_beam_sgc,
                        "lnll": final_response_beam_sgc_lnll.item(),
                    }
                )

            # Store the correctness for the current normalization type
            metrics[norm_type]["beam_sgc_answer_correctness"] = current_correctness

        ###################################################################
        # Sampling evaluation
        ###################################################################
        # Loop over different numbers of generations
        for n_gen in generation_numbers:
            gen_kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
                "return_dict_in_generate": True,
                "pad_token_id": tokenizer.eos_token_id,
                "num_return_sequences": n_gen,  # Generate n_gen samples
            }
            
            gen_kwargs["temperature"] = 1.0 if temperature_head else temperature
                
            if top_p:
                gen_kwargs["top_p"] = 0.9
                
            if top_k:
                gen_kwargs["top_k"] = 50
                            
            with t.no_grad():
                outputs = model.generate(
                    **tokenized_inputs,
                    **gen_kwargs
                )

            decoded_outputs = tokenizer.batch_decode(
                outputs.sequences[:, input_len:], skip_special_tokens=True
            )
            
            decoded_outputs = [clean_generated_text(
                text) for text in decoded_outputs]
                                    
            for y, prompt in enumerate(prompts):
                start_idx = y * n_gen
                end_idx = start_idx + n_gen

                individual_decoded_outputs = decoded_outputs[start_idx:end_idx]
                
                # Include all generations, including duplicates
                generated_texts = individual_decoded_outputs  # *** Use generated_texts directly

                (
                clusters,
                num_clusters,
                cluster_examples_dict,
                cluster_lls,
                cluster_normalized_lls,
                ) = compute_clusters_and_lls(
                    generated_texts=generated_texts,
                    prompt=prompt,
                    args=args,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    temperature=temperature,
                    temperature_head=temperature_head,
                    model_id=model_id,
                    support_examples=support_examples,
                    entailment_model=entailment_model,
                    entailment_tokenizer=entailment_tokenizer,
                    pipe=pipe,
                    use_modernbert=use_modernbert,
                    nli_device=nli_device,
                )
                

                for norm_type in metrics:
                    cluster_lls_to_use = cluster_normalized_lls if norm_type == "normalised" else cluster_lls
                    
                    ###################################################################
                    # ML-MGC evaluation
                    ###################################################################

                    # Compute log-sum-exp of log-likelihoods for each cluster
                    cluster_log_sums = t.tensor([
                        t.logsumexp(cluster_ll, dim=0) for cluster_ll in cluster_lls_to_use.values()
                    ]).to(device)

                    # Compute log of the average likelihood for each cluster
                    cluster_sizes = t.tensor(
                        [len(cluster_ll)
                         for cluster_ll in cluster_lls_to_use.values()],
                        dtype=t.float32
                    ).to(device)
                    log_cluster_avg_likelihoods = cluster_log_sums - \
                        t.log(cluster_sizes)

                    # Apply softmax over log-average-likelihoods to get cluster probabilities
                    log_cluster_probs = log_cluster_avg_likelihoods - \
                        t.logsumexp(log_cluster_avg_likelihoods, dim=0)
                    cluster_distribution_ml_mgc = t.exp(log_cluster_probs)

                    tag = "ml_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=cluster_distribution_ml_mgc,
                        cluster_lls_map=cluster_lls_to_use,       
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )


                    ###################################################################
                    # H-SC (exp‐penalty) evaluation
                    ###################################################################


                    # Compute length‐normalized probabilities (SGC) for each response in each cluster
                    log_sgc_clusters = {
                        key: lls  # lls are length‐normalized log‐likelihoods
                        for key, lls in cluster_lls_to_use.items()
                    }

                    # Prepare lists to collect per‐cluster mean SGC and entropy
                    cluster_keys = list(log_sgc_clusters.keys())
                    log_mean_sgc_list = []
                    entropy_list = []

                    eps = 1e-12  # small constant to avoid log(0)
                    for key in cluster_keys:
                        log_sgc_vals   = log_sgc_clusters[key]          
                        total_log_sgc  = t.logsumexp(log_sgc_vals, dim=0)  

                        # 1) true log-probs
                        log_q_i = log_sgc_vals - total_log_sgc           

                        # 2) floor them so you never see -∞
                        log_q_i = t.clamp(log_q_i, min=t.log(t.tensor(eps, device=log_q_i.device)))

                        # 3) get back to probs and renormalize
                        q_i = t.exp(log_q_i)
                        q_i = q_i / q_i.sum()

                        # 4) entropy with your clamped log-probs
                        entropy = -(q_i * log_q_i).sum()  
                        entropy_list.append(entropy)
                        
                        n_t = t.tensor(len(log_sgc_vals), dtype=total_log_sgc.dtype, device=total_log_sgc.device)
                        
                        mean_log_sgc = total_log_sgc - t.log(n_t)  if len(log_sgc_vals) > 0 else 0.0
                        log_mean_sgc_list.append(mean_log_sgc)  # scalar

                    # Stack into tensors on the same device
                    log_mean_sgc_tensor = t.stack(log_mean_sgc_list).to(device)    # shape: (k,)
                    entropy_tensor  = t.stack(entropy_list).to(device)     # shape: (k,)
                    
                    # Compute unnormalized H-SC scores using multiplicative exp‐penalty: mean_sgc * exp(-entropy)
                    log_h_sc = log_mean_sgc_tensor - entropy_tensor 
                    log_h_sc_probs = log_h_sc - t.logsumexp(log_h_sc, dim=0)  # shape: (k,)
                    h_sc_probs = t.exp(log_h_sc_probs)  # shape: (k,)


                    tag = "h_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=h_sc_probs,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    ###################################################################
                    # H-SC (Unnormalized) evaluation
                    ###################################################################

                    # Prepare lists to collect per‐cluster mean SGC and entropy
                    log_sum_sgc_list = []
                    entropy_list = []

                    eps = 1e-12  # small constant to avoid log(0)
                    for key in cluster_keys:
                        log_sgc_vals   = log_sgc_clusters[key]          
                        total_log_sgc  = t.logsumexp(log_sgc_vals, dim=0)  

                        # 1) true log-probs
                        log_q_i = log_sgc_vals - total_log_sgc           

                        # 2) floor them so you never see -∞
                        log_q_i = t.clamp(log_q_i, min=t.log(t.tensor(eps, device=log_q_i.device)))

                        # 3) get back to probs and renormalize
                        q_i = t.exp(log_q_i)
                        q_i = q_i / q_i.sum()

                        # 4) entropy with your clamped log-probs
                        entropy = -(q_i * log_q_i).sum()  
                        entropy_list.append(entropy)
                        
                        log_sum_sgc_list.append(total_log_sgc)  # scalar

                    # Stack into tensors on the same device
                    log_sum_sgc_tensor = t.stack(log_sum_sgc_list).to(device)    # shape: (k,)
                    entropy_tensor  = t.stack(entropy_list).to(device)     # shape: (k,)
                    
                    # Compute unnormalized H-SC scores using multiplicative exp‐penalty: mean_sgc * exp(-entropy)
                    log_h_sc_unnorm = log_sum_sgc_tensor - entropy_tensor 
                    log_h_sc_unorm_probs = log_h_sc_unnorm - t.logsumexp(log_h_sc_unnorm, dim=0)  # shape: (k,)
                    h_sc_unorm_probs = t.exp(log_h_sc_unorm_probs)  # shape: (k,)


                    # 6) Delegate to the generic helper
                    tag = "h_unnorm_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=h_sc_unorm_probs,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=eps,
                    )

                    ###################################################################
                    # MGC evaluation
                    ###################################################################

                    # Calculate cluster counts based on cluster size
                    cluster_counts = t.tensor(
                        [len(cluster) for cluster in clusters]).to(device)

                    if cluster_counts.sum() != 0:
                        cluster_distribution_mgc = cluster_counts / cluster_counts.sum()
                    else:
                        print("Error here")
                        print(cluster_counts)
                        continue

                    tag = "mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=cluster_distribution_mgc,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    # #-------------------------------------------------------------------
                    # # Exp-MGC evaluation (softmax over counts)
                    # #-------------------------------------------------------------------
                    # # cluster_counts already computed above for MGC:
                    # #   cluster_counts = tensor([len(cluster) for cluster in clusters], device=device)
                    # exp_cluster_probs = t.softmax(cluster_counts.float(), dim=0)
                    # tag = "exp_mgc"
                    # evaluate_one_cluster_method(
                    #     tag=tag,
                    #     cluster_scores=exp_cluster_probs,
                    #     cluster_lls_map=cluster_lls_to_use,
                    #     cluster_examples_dict=cluster_examples_dict,
                    #     refs=cleaned_gold_lists,
                    #     classes=classes,
                    #     y=y,
                    #     prompt=prompt,
                    #     beam_response=beam_sgc_decoded_outputs[y],
                    #     norm_type=norm_type,
                    #     n_gen=n_gen,
                    #     num_clusters=num_clusters,
                    #     metrics=metrics,
                    #     args_dset=args.dset,
                    #     eps=1e-12,
                    # )

                    ###################################################################
                    # L-MGC evaluation
                    ###################################################################

                    # Compute log-sum-exp of log-likelihoods for each cluster
                    cluster_log_sums = t.tensor([
                        t.logsumexp(cluster_ll, dim=0) for cluster_ll in cluster_lls_to_use.values()
                    ]).to(device)

                    # Apply softmax over the log-sums to get cluster probabilities
                    cluster_distribution_l_mgc = t.softmax(
                        cluster_log_sums, dim=0)

                    tag = "l_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=cluster_distribution_l_mgc,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )

                    ###################################################################
                    # B-MGC evaluation
                    ###################################################################

                    # Sum the log-likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihood_sum = t.sum(
                            cluster_lls_to_use[cluster_key])
                        cluster_likelihoods.append(cluster_likelihood_sum)

                    # Convert cluster_likelihoods to a tensor
                    cluster_likelihoods = t.stack(
                        cluster_likelihoods).to(device)

                    # Calculate the prior and posterior distributions
                    log_prior_distribution = t.log(
                        cluster_counts.float() / cluster_counts.sum() + 1e-12)
                    
                    log_unnormalised_posterior = log_prior_distribution + cluster_likelihoods
                    log_posterior_distribution = log_unnormalised_posterior - \
                        t.logsumexp(log_unnormalised_posterior, dim=-1)
                    posterior_distribution = log_posterior_distribution.exp()

                    tag = "b_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior_distribution,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )

                    ###################################################################
                    # Gibbs_0.75_MGC evaluation
                    ###################################################################

                    alpha = 0.75

                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihood_sum = t.sum(cluster_lls_to_use[cluster_key])
                        cluster_likelihoods.append(cluster_likelihood_sum)

                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)

                    # Prior in log‐space
                    log_prior_distribution = t.log(cluster_counts.float() / cluster_counts.sum() + 1e-12)

                    # Tempered (Gibbs) posterior: log p ∝ log prior + α * log-likelihood
                    log_unnormalised_gibbs = log_prior_distribution + alpha * cluster_likelihoods
                    log_posterior_distribution = log_unnormalised_gibbs - t.logsumexp(log_unnormalised_gibbs, dim=-1)
                    posterior_distribution = log_posterior_distribution.exp()

                    # Tag and evaluate
                    
                    # Replace . with _ here for the string version of alpha
                    alpha_str = str(alpha).replace(".", "_")
                    tag = f"gibbs_{alpha_str}_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior_distribution,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    ###################################################################
                    # Gibbs_0.5_MGC evaluation
                    ###################################################################

                    alpha = 0.5

                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihood_sum = t.sum(cluster_lls_to_use[cluster_key])
                        cluster_likelihoods.append(cluster_likelihood_sum)

                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)

                    # Prior in log‐space
                    log_prior_distribution = t.log(cluster_counts.float() / cluster_counts.sum() + 1e-12)

                    # Tempered (Gibbs) posterior: log p ∝ log prior + α * log-likelihood
                    log_unnormalised_gibbs = log_prior_distribution + alpha * cluster_likelihoods
                    log_posterior_distribution = log_unnormalised_gibbs - t.logsumexp(log_unnormalised_gibbs, dim=-1)
                    posterior_distribution = log_posterior_distribution.exp()

                    # Tag and evaluate
                    alpha_str = str(alpha).replace(".", "_")
                    tag = f"gibbs_{alpha_str}_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior_distribution,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    ###################################################################
                    # Gibbs_1.25_MGC evaluation
                    ###################################################################

                    alpha = 1.25

                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihood_sum = t.sum(cluster_lls_to_use[cluster_key])
                        cluster_likelihoods.append(cluster_likelihood_sum)

                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)

                    # Prior in log‐space
                    log_prior_distribution = t.log(cluster_counts.float() / cluster_counts.sum() + 1e-12)

                    # Tempered (Gibbs) posterior: log p ∝ log prior + α * log-likelihood
                    log_unnormalised_gibbs = log_prior_distribution + alpha * cluster_likelihoods
                    log_posterior_distribution = log_unnormalised_gibbs - t.logsumexp(log_unnormalised_gibbs, dim=-1)
                    posterior_distribution = log_posterior_distribution.exp()

                    # Tag and evaluate
                    alpha_str = str(alpha).replace(".", "_")
                    tag = f"gibbs_{alpha_str}_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior_distribution,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    # ###################################################################
                    # # B_exp_MGC evaluation (γ = 1)
                    # ###################################################################

                    # # Exponential prior weight γ
                    # gamma = 1.0

                    # # Sum the log‐likelihoods for each cluster
                    # cluster_likelihoods = []
                    # for cluster_key in cluster_lls_to_use:
                    #     cluster_likelihood_sum = t.sum(cluster_lls_to_use[cluster_key])
                    #     cluster_likelihoods.append(cluster_likelihood_sum)

                    # cluster_likelihoods = t.stack(cluster_likelihoods).to(device)  # shape (K,)

                    # # Compute exponential prior in log‐space: log π_c = γ * N_c
                    # log_prior_distribution = gamma * cluster_counts.float()  # shape (K,)

                    # # Unnormalised log‐posterior: log π_c + ℓ_c
                    # log_unnormalised_posterior = log_prior_distribution + cluster_likelihoods

                    # # Normalise to get log p(c | D)
                    # log_posterior_distribution = log_unnormalised_posterior - t.logsumexp(
                    #     log_unnormalised_posterior, dim=-1
                    # )

                    # # Exponentiate to get p(c | D)
                    # posterior_distribution = log_posterior_distribution.exp()

                    # # Tag and evaluate
                    # tag = "b_exp_mgc"
                    # evaluate_one_cluster_method(
                    #     tag=tag,
                    #     cluster_scores=posterior_distribution,
                    #     cluster_lls_map=cluster_lls_to_use,
                    #     cluster_examples_dict=cluster_examples_dict,
                    #     refs=cleaned_gold_lists,
                    #     classes=classes,
                    #     y=y,
                    #     prompt=prompt,
                    #     beam_response = beam_sgc_decoded_outputs[y],
                    #     norm_type=norm_type,
                    #     n_gen=n_gen,
                    #     num_clusters=num_clusters,
                    #     metrics=metrics,
                    #     args_dset=args.dset,
                    #     eps=1e-12,
                    # )
                    
                    ###################################################################
                    # Median-SC evaluation
                    ###################################################################

                    # Compute length‐normalized probabilities (SGC) for each response in each cluster
                    sgc_clusters = {
                        key: t.exp(lls)  # lls are length‐normalized log‐likelihoods
                        for key, lls in cluster_lls_to_use.items()
                    }

                    # 2) Collect medians
                    median_sgc_list = [
                        t.median(sgc_clusters[key]) 
                        for key in sgc_clusters
                    ]

                    # 3) Stack into a single tensor (stays on GPU)
                    median_sgc_tensor = t.stack(median_sgc_list, dim=0)  # shape: (K,)

                    # 4) Smooth and normalize
                    eps = 1e-12
                    K   = median_sgc_tensor.size(0)
                    median_sc_probs = (median_sgc_tensor + eps) \
                                    / (median_sgc_tensor.sum() + eps * K)  # shape: (K,)

                    tag = "median_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=median_sc_probs,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )

                    # ###################################################################
                    # # Geometric-Mean SC evaluation
                    # ###################################################################


                    # # Compute length‐normalized probabilities (SGC) for each response in each cluster
                    # sgc_clusters = {
                    #     key: t.exp(lls)  # lls are length‐normalized log‐likelihoods
                    #     for key, lls in cluster_lls_to_use.items()
                    # }

                    # # Prepare to collect (geometric mean of SGC) in each cluster
                    # cluster_keys = list(sgc_clusters.keys())
                    # gmean_sgc_list = []

                    # eps = 1e-12  # to avoid log(0) inside the geometric mean
                    # for key in cluster_keys:
                    #     sgc_vals = sgc_clusters[key]  # shape: (|C_i|,)

                    #     # 1) Geometric mean of all SGC values in this cluster:
                    #     #      gmean = exp(mean(log(sgc_vals + eps)))
                    #     gmean = t.exp(t.log(sgc_vals + eps).mean())

                    #     gmean_sgc_list.append(gmean)

                    # # Stack into tensor on the same device
                    # gmean_sgc_tensor = t.stack(gmean_sgc_list).to(device)  # shape: (k,)

                    # # 2) Normalize across clusters to get a valid “geometric SC” distribution
                    # gmean_sc_probs = gmean_sgc_tensor / (gmean_sgc_tensor.sum() + eps)  # shape: (k,)

                    # # 3) Delegate to the generic helper
                    # tag = "geometric_mean_mgc"
                    # evaluate_one_cluster_method(
                    #     tag=tag,
                    #     cluster_scores=gmean_sc_probs,
                    #     cluster_lls_map=cluster_lls_to_use,
                    #     cluster_examples_dict=cluster_examples_dict,
                    #     refs=cleaned_gold_lists,
                    #     classes=classes,
                    #     y=y,
                    #     prompt=prompt,
                    #     beam_response = beam_sgc_decoded_outputs[y],
                    #     norm_type=norm_type,
                    #     n_gen=n_gen,
                    #     num_clusters=num_clusters,
                    #     metrics=metrics,
                    #     args_dset=args.dset,
                    #     eps=eps,
                    # )

                    # ###################################################################
                    # # Mean-SC evaluation
                    # ###################################################################

                    # # Collect mean SGC per cluster
                    # mean_sgc_list = []
                    # for key in cluster_keys:
                    #     sgc_vals = sgc_clusters[key]
                    #     mean_sgc = sgc_vals.mean()               # scalar
                    #     mean_sgc_list.append(mean_sgc)

                    # # Stack into tensor
                    # mean_sgc_tensor = t.stack(mean_sgc_list).to(device)  # shape: (k,)

                    # # Normalize across clusters to get Mean-SC distribution
                    # mean_sc_unnorm = mean_sgc_tensor
                    # mean_sc_probs = mean_sc_unnorm / (mean_sc_unnorm.sum())

                    # tag = "mean_mgc"
                    # evaluate_one_cluster_method(
                    #     tag=tag,
                    #     cluster_scores=mean_sc_probs,
                    #     cluster_lls_map=cluster_lls_to_use,
                    #     cluster_examples_dict=cluster_examples_dict,
                    #     refs=cleaned_gold_lists,
                    #     classes=classes,
                    #     y=y,
                    #     prompt=prompt,
                    #     beam_response = beam_sgc_decoded_outputs[y],
                    #     norm_type=norm_type,
                    #     n_gen=n_gen,
                    #     num_clusters=num_clusters,
                    #     metrics=metrics,
                    #     args_dset=args.dset,
                    #     eps=1e-12,
                    # )

                    ###################################################################
                    # Min-SC evaluation
                    ###################################################################


                    # Compute length‐normalized probabilities (SGC) for each response in each cluster

                    # Collect min SGC per cluster
                    cluster_keys = list(sgc_clusters.keys())
                    min_sgc_list = []
                    for key in cluster_keys:
                        sgc_vals = sgc_clusters[key]    # shape: (|C_i|,)
                        min_sgc = sgc_vals.min()        # scalar
                        min_sgc_list.append(min_sgc)

                    # Stack into tensor on the same device
                    min_sgc_tensor = t.stack(min_sgc_list).to(device)  # shape: (k,)

                    # Normalize across clusters to get Min-SC distribution
                    eps = 1e-12
                    min_sc_probs = min_sgc_tensor / (min_sgc_tensor.sum() + eps)  # shape: (k,)

                    tag = "min_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=min_sc_probs,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=eps,
                    )

                    ###################################################################
                    # Cold-0.75-MGC evaluation (T = 0.75)
                    ###################################################################

                    T = 0.75


                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihoods.append(
                            t.sum(cluster_lls_to_use[cluster_key])
                        )
                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)  # shape (K,)

                    # Compute log‐prior = log N_c − log sum N_d
                    cluster_counts = t.tensor(
                        [len(cluster) for cluster in clusters],
                        dtype=t.float32
                    ).to(device)
                    log_prior = t.log(cluster_counts / cluster_counts.sum() + 1e-12)

                    # Temperature‐scaled log‐posterior: (log_prior + ℓ_c) / T
                    log_unnormalised = (log_prior + cluster_likelihoods) / T
                    log_posterior = log_unnormalised - t.logsumexp(log_unnormalised, dim=-1)
                    posterior = log_posterior.exp()

                    tag = "cold_0_75_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )

                    ###################################################################
                    # Cold-0.5-MGC evaluation (T = 0.5)
                    ###################################################################

                    T = 0.5


                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihoods.append(
                            t.sum(cluster_lls_to_use[cluster_key])
                        )
                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)

                    # Compute log‐prior
                    log_prior = t.log(cluster_counts / cluster_counts.sum() + 1e-12)

                    # Temperature‐scaled log‐posterior
                    log_unnormalised = (log_prior + cluster_likelihoods) / T
                    log_posterior = log_unnormalised - t.logsumexp(log_unnormalised, dim=-1)
                    posterior = log_posterior.exp()

                    tag = "cold_0_5_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )

                    ###################################################################
                    # Cold-1.25-MGC evaluation (T = 1.25)
                    ###################################################################

                    T = 1.25


                    # Sum the log‐likelihoods for each cluster
                    cluster_likelihoods = []
                    for cluster_key in cluster_lls_to_use:
                        cluster_likelihoods.append(
                            t.sum(cluster_lls_to_use[cluster_key])
                        )
                    cluster_likelihoods = t.stack(cluster_likelihoods).to(device)

                    # Compute log‐prior
                    log_prior = t.log(cluster_counts / cluster_counts.sum() + 1e-12)

                    # Temperature‐scaled log‐posterior
                    log_unnormalised = (log_prior + cluster_likelihoods) / T
                    log_posterior = log_unnormalised - t.logsumexp(log_unnormalised, dim=-1)
                    posterior = log_posterior.exp()

                    tag = "cold_1_25_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=posterior,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    # ###################################################################
                    # # Dirichlet-MGC evaluation (Laplace smoothing with β = 1.0)
                    # ###################################################################

                    # # Smoothing parameter β
                    # beta = 1.0

                    # # Compute raw cluster counts
                    # cluster_counts = t.tensor(
                    #     [len(cluster) for cluster in clusters],
                    #     dtype=t.float32
                    # ).to(device)  # shape: (K,)

                    # # Apply Dirichlet (Laplace) smoothing: N_c + β
                    # smoothed_counts = cluster_counts + beta  # shape: (K,)

                    # # Normalize to get a proper distribution
                    # cluster_distribution_dirichlet = smoothed_counts / smoothed_counts.sum()

                    # tag = "dirichlet_mgc"
                    # evaluate_one_cluster_method(
                    #     tag=tag,
                    #     cluster_scores=cluster_distribution_dirichlet,
                    #     cluster_lls_map=cluster_lls_to_use,
                    #     cluster_examples_dict=cluster_examples_dict,
                    #     refs=cleaned_gold_lists,
                    #     classes=classes,
                    #     y=y,
                    #     prompt=prompt,
                    #     beam_response = beam_sgc_decoded_outputs[y],
                    #     norm_type=norm_type,
                    #     n_gen=n_gen,
                    #     num_clusters=num_clusters,
                    #     metrics=metrics,
                    #     args_dset=args.dset,
                    #     eps=1e-12,
                    # )

                    ###################################################################
                    # Max-SC evaluation
                    ###################################################################

                    # Collect max SGC per cluster
                    log_maxes = t.stack(
                    [ cluster_lls_to_use[k].max() for k in cluster_lls_to_use ],
                    dim=0
                    )               # avoids the extra CPU↔GPU hop
                    log_p      = log_maxes - t.logsumexp(log_maxes, dim=0)
                    max_sc_probs = t.exp(log_p)

                    tag = "max_mgc"
                    evaluate_one_cluster_method(
                        tag=tag,
                        cluster_scores=max_sc_probs,
                        cluster_lls_map=cluster_lls_to_use,
                        cluster_examples_dict=cluster_examples_dict,
                        refs=cleaned_gold_lists,
                        classes=classes,
                        y=y,
                        prompt=prompt,
                        beam_response = beam_sgc_decoded_outputs[y],
                        norm_type=norm_type,
                        n_gen=n_gen,
                        num_clusters=num_clusters,
                        metrics=metrics,
                        args_dset=args.dset,
                        eps=1e-12,
                    )
                    
                    ###################################################################
                    
    # Initialize `overall_results`
    overall_results = {"normalised": {}}
    rejection_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
            0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
            0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
        
    # Loop over both normalization types and bind metrics
    for norm_type, norm_metrics in metrics.items():
        overall_results[norm_type] = {}
        overall_results[norm_type]["beam_sgc"] = {}

            
        # Beam SGC metrics calculation and storage
        nll_beam_sgc = norm_metrics["total_nll_beam_sgc"] / num_examples
        nlc_beam_sgc = norm_metrics["total_nlc_beam_sgc"] / num_examples
        acc_beam_sgc = norm_metrics["accuracy_beam_sgc"] / num_examples
        # Compute additional metrics as in your original structure
        ece_beam_sgc_conf = compute_ece(
            norm_metrics["ece_values_beam_sgc_conf"], num_bins=args.num_bins)
        ace_beam_sgc_conf = compute_ace(
            norm_metrics["ece_values_beam_sgc_conf"], num_bins=args.num_bins)
        corp_mcb, corp_dsc, corp_unc, corp_orig_brier, corp_recal_brier = compute_corp_decomposition(
            norm_metrics["ece_values_beam_sgc_conf"])
        prr_beam_sgc_conf = compute_prr(
            norm_metrics["prr_values_beam_sgc_conf"])
        auroc_beam_sgc_conf = compute_auroc(
            norm_metrics["auroc_values_beam_sgc_conf"], uncertainty=False)
        aurac_beam_sgc_conf = compute_aurac(
            norm_metrics["auroc_values_beam_sgc_conf"], uncertainty=False)
        
        
        sel_aurocs = {}
        sel_rej_accs = {}

        for r in rejection_rates:
            # call your two functions once each
            auroc = compute_selective_auroc(
                norm_metrics["auroc_values_beam_sgc_conf"],
                rejection_rate=r,
                uncertainty=False
            )
            acc = compute_rejection_accuracy(
                norm_metrics["auroc_values_beam_sgc_conf"],
                rejection_rate=r,
                uncertainty=False
            )

            # storing them in dicts keyed by an int version of r*100
            key = int(r * 100)  # e.g. 1, 5, 10, ...
            sel_aurocs[f"SEL_AUROC_{key}_conf"] = auroc
            sel_rej_accs[f"SEL_REJECTION_{key}_conf"] = acc
        
        brier_beam_sgc_conf = norm_metrics["total_brier_beam_sgc_conf"] / num_examples
        

        # Store computed beam SGC metrics including entropies
        overall_results[norm_type]["beam_sgc"].update({
            "ACC": acc_beam_sgc,
            "NLL": nll_beam_sgc,
            "NLC": nlc_beam_sgc,
            "PRR_conf": prr_beam_sgc_conf,
            "ECE_conf": ece_beam_sgc_conf,
            "ACE_conf": ace_beam_sgc_conf,
            "CORP_MCB": corp_mcb,
            "CORP_Brier": corp_orig_brier,
            "CORP_Recal_Brier": corp_recal_brier,
            "CORP_DSC": corp_dsc,
            "CORP_Unc": corp_unc,
            "Brier_conf": brier_beam_sgc_conf,
            "AUROC_conf": auroc_beam_sgc_conf,
            "AURAC_conf": aurac_beam_sgc_conf,     
        })

        
        overall_results[norm_type]["beam_sgc"].update(sel_aurocs)
        overall_results[norm_type]["beam_sgc"].update(sel_rej_accs)

    # After initialization, populate metrics within evaluation
    # Loop over each `n_gen` and calculate/store MGC, ML_MGC, etc. metrics
    for n_gen in generation_numbers:
        for norm_type, norm_metrics in metrics.items():
            for mgc_type in [
                    "MGC","ML_MGC","B_MGC","L_MGC",
                    "MEDIAN_MGC","MAX_MGC",
                    "H_MGC", "H_UNNORM_MGC","COLD_0_75_MGC","COLD_0_5_MGC","COLD_1_25_MGC",
                    "GIBBS_0_75_MGC","GIBBS_0_5_MGC","GIBBS_1_25_MGC", "MIN_MGC"]:
                
                print(f"Processing {mgc_type} for n_gen={n_gen} and norm_type={norm_type}")
                
                mgc_key = f"{mgc_type}_n{n_gen}"

                nll_mgc = norm_metrics[f"total_nll_{mgc_type.lower()}"][n_gen] / \
                    num_examples
                nlc_mgc = norm_metrics[f"total_nlc_{mgc_type.lower()}"][n_gen] / \
                    num_examples
                acc_mgc = norm_metrics[f"accuracy_{mgc_type.lower()}"][n_gen] / \
                    num_examples
                    
                # Compute additional metrics for entropy, AUROC, ECE, Brier, etc.
                ece_mgc_conf = compute_ece(
                    norm_metrics[f"ece_values_{mgc_type.lower()}_conf"][n_gen], num_bins=args.num_bins)
                ace_mgc_conf = compute_ace(
                    norm_metrics[f"ece_values_{mgc_type.lower()}_conf"][n_gen], num_bins=args.num_bins
                )
                corp_mcb, corp_dsc, corp_unc, corp_orig_brier, corp_recal_brier = compute_corp_decomposition(
                    norm_metrics[f"ece_values_{mgc_type.lower()}_conf"][n_gen]
                )
                prr_mgc_conf = compute_prr(
                    norm_metrics[f"prr_values_{mgc_type.lower()}_conf"][n_gen])
                auroc_mgc_conf = compute_auroc(
                    norm_metrics[f"auroc_values_{mgc_type.lower()}_conf"][n_gen], uncertainty=False)
                aurac_mgc_conf = compute_aurac(
                    norm_metrics[f"auroc_values_{mgc_type.lower()}_conf"][n_gen], uncertainty=False)
                brier_mgc_conf = norm_metrics[f"total_brier_{mgc_type.lower()}_conf"][n_gen] / \
                    num_examples
                    
                sel_aurocs = {}
                sel_rej_accs = {}

                for r in rejection_rates:
                    # call your two functions once each
                    auroc = compute_selective_auroc(
                        norm_metrics[f"auroc_values_{mgc_type.lower()}_conf"][n_gen],
                        rejection_rate=r,
                        uncertainty=False
                    )
                    acc = compute_rejection_accuracy(
                        norm_metrics[f"auroc_values_{mgc_type.lower()}_conf"][n_gen],
                        rejection_rate=r,
                        uncertainty=False
                    )

                    # storing them in dicts keyed by an int version of r*100
                    key = int(r * 100)  # e.g. 1, 5, 10, ...
                    sel_aurocs[f"SEL_AUROC_{key}_conf"] = auroc
                    sel_rej_accs[f"SEL_REJECTION_{key}_conf"] = acc
                    
                # Create the dictionary so that we can store results.
                if mgc_key not in overall_results[norm_type]:
                    overall_results[norm_type][mgc_key] = {}
                    
                # Compute the auroc of the different entropies.
                auroc_entropy_rb = compute_auroc(
                    norm_metrics[f"{mgc_type.lower()}_rb_entropy_values"][n_gen], uncertainty=True)
                
                aurac_entropy_rb = compute_aurac(
                    norm_metrics[f"{mgc_type.lower()}_rb_entropy_values"][n_gen], uncertainty = True),
                
                auroc_entropy_mc = compute_auroc(
                    norm_metrics[f"{mgc_type.lower()}_mc_entropy_values"][n_gen], uncertainty=True)
                aurac_entropy_mc = compute_aurac(
                    norm_metrics[f"{mgc_type.lower()}_mc_entropy_values"][n_gen], uncertainty=True)
                
                # Compute the acutal semantic entropy values too.
                auroc_entropy_se_rb = compute_auroc(
                    norm_metrics[f"{mgc_type.lower()}_rb_entropy_values_SE"][n_gen], uncertainty=True)
                aurac_entropy_se_rb = compute_aurac(
                    norm_metrics[f"{mgc_type.lower()}_rb_entropy_values_SE"][n_gen], uncertainty=True)

                auroc_entropy_se_mc = compute_auroc(
                    norm_metrics[f"{mgc_type.lower()}_mc_entropy_values_SE"][n_gen], uncertainty=True)
                aurac_entropy_se_mc = compute_aurac(
                    norm_metrics[f"{mgc_type.lower()}_mc_entropy_values_SE"][n_gen], uncertainty=True)
                
                
                # Store MGC metrics under the current normalization type and generation number
                overall_results[norm_type][mgc_key] = {
                    "ACC": acc_mgc,
                    "NLL": nll_mgc,
                    "NLC": nlc_mgc,
                    "PRR_conf": prr_mgc_conf,
                    "ECE_conf": ece_mgc_conf,
                    "ACE_conf": ace_mgc_conf,
                    "CORP_MCB": corp_mcb,
                    "CORP_Brier": corp_orig_brier,
                    "CORP_Recal_Brier": corp_recal_brier,
                    "CORP_DSC": corp_dsc,
                    "CORP_Unc": corp_unc,
                    "Brier_conf": brier_mgc_conf,
                    "AUROC_conf": auroc_mgc_conf,
                    "AURAC_conf": aurac_mgc_conf,
                    "AUROC_entropy_rb": auroc_entropy_rb,
                    "AUROC_entropy_se_rb": auroc_entropy_se_rb,
                    "AUROC_entropy_mc": auroc_entropy_mc,
                    "AUROC_entropy_se_mc": auroc_entropy_se_mc,
                    "AURAC_entropy_rb": aurac_entropy_rb,
                    "AURAC_entropy_se_rb": aurac_entropy_se_rb,
                    "AURAC_entropy_mc": aurac_entropy_mc,
                    "AURAC_entropy_se_mc": aurac_entropy_se_mc,
                }
                
                overall_results[norm_type][mgc_key].update(sel_aurocs)
                overall_results[norm_type][mgc_key].update(sel_rej_accs)

    return {
        "overall_results": overall_results,
        "sgc_results": {
            "normalised": metrics["normalised"]["beam_sgc_results"]
        },
        "e_sc_results": {
            "normalised": metrics["normalised"]["mgc_results"]
        },
        # "exp_sc_results": {
        #     "normalised": metrics["normalised"]["exp_mgc_results"]
        # },
        "l_sc_results": {
            "normalised": metrics["normalised"]["l_mgc_results"]
        },
        "ml_sc_results": {
            "normalised": metrics["normalised"]["ml_mgc_results"]
        },
        "h_norm_sc_results": {
            "normalised": metrics["normalised"]["h_mgc_results"]
        },
        "h_unnorm_sc_results": {
            "normalised": metrics["normalised"]["h_unnorm_mgc_results"]
        },
        "b_sc_results": {
            "normalised": metrics["normalised"]["b_mgc_results"]
        },
        "median_sc_results": {"normalised": metrics["normalised"]["median_mgc_results"]},
        # "arithmetic_mean_sc_results":   {"normalised": metrics["normalised"]["mean_mgc_results"]},
        # "geometric_mean_sc_results": {"normalised": metrics["normalised"]["geometric_mean_mgc_results"]},
        "max_sc_results":    {"normalised": metrics["normalised"]["max_mgc_results"]},
        "gibbs_0_5_sc_results": {"normalised": metrics["normalised"]["gibbs_0_5_mgc_results"]},
        "gibbs_0_75_sc_results": {"normalised": metrics["normalised"]["gibbs_0_75_mgc_results"]},
        "gibbs_1_25_sc_results": {"normalised": metrics["normalised"]["gibbs_1_25_mgc_results"]},
        "min_sc_results": {"normalised": metrics["normalised"]["min_mgc_results"]},
        # "dirichlet_sc_results": {"normalised": metrics["normalised"]["dirichlet_mgc_results"]},
        # "b_exp_sc_results": {"normalised": metrics["normalised"]["b_exp_mgc_results"]},
        "cold_1_25_sc_results": {"normalised": metrics["normalised"]["cold_1_25_mgc_results"]},
        "cold_0_75_sc_results": {"normalised": metrics["normalised"]["cold_0_75_mgc_results"]},
        "cold_0_5_sc_results": {"normalised": metrics["normalised"]["cold_0_5_mgc_results"]},
    }

def evaluate_model_and_save_results(
    save_path,
    args,
    model,
    model_id,
    tokenizer,
    calib_validation,
    final_test,
    temperature,
    entailment_model,
    entailment_tokenizer,
    llm_device,
    nli_device,
    support_examples=None,
    temperature_head = False
):
    
    # Initialize aggregated results for normalized and unnormalized metrics
    # We'll store these under the new keys: "validation_train", "validation_test", and "test".
    aggregated_results = {
        "normalised": {
            "calibration_validation": {},
            "test": {},
        },
    }

    runs_save_path = os.path.join(save_path, "runs")
    os.makedirs(runs_save_path, exist_ok=True)

    for repeat in range(args.num_repeats):
        current_seed = args.seed + repeat
        set_seed(current_seed)

        # Create a directory for this particular run within the 'runs' folder
        repeat_save_path = os.path.join(runs_save_path, f"run_{repeat + 1}")
        os.makedirs(repeat_save_path, exist_ok=True)

        model.eval()

        # ------------------------------------------------------------------------------------------
        # Evaluate on validation train split (the 80%)
        # ------------------------------------------------------------------------------------------
        print(f"Evaluating model on the calibration validation split (run {repeat+1})...")
        calib_validation_results = evaluate_model(
            model,
            model_id,
            tokenizer,
            calib_validation,
            temperature,
            args,
            entailment_model,
            entailment_tokenizer,
            device=llm_device,
            nli_device=nli_device,
            generation_numbers=args.generation_numbers,
            support_examples = support_examples,
            temperature_head = temperature_head
        )

        # ------------------------------------------------------------------------------------------
        # Evaluate on the final test split
        # ------------------------------------------------------------------------------------------
        print(f"Evaluating model on the final test split (run {repeat+1})...")
        test_results = evaluate_model(
            model,
            model_id,
            tokenizer,
            final_test,
            temperature,
            args,
            entailment_model,
            entailment_tokenizer,
            device=llm_device,
            nli_device=nli_device,
            generation_numbers=args.generation_numbers,
            support_examples = support_examples,
            temperature_head = temperature_head
        )

        # Process and save normalized and unnormalized results for this run
        for norm_type in ["normalised"]:
            print(f"[Run {repeat+1}] Processing & saving {norm_type} results...")

            norm_save_path = os.path.join(repeat_save_path, norm_type)
            os.makedirs(norm_save_path, exist_ok=True)

            # Extract the top-level dictionaries for each subset
            overall_calib_validation_results = calib_validation_results["overall_results"][norm_type]

            overall_test_results = test_results["overall_results"][norm_type]
            sgc_test_results = test_results["sgc_results"][norm_type]

            # The sampling code typically uses generation_numbers = [10], etc.
            # Make sure these match what you used in your evaluate_model

            for n_gen in args.generation_numbers:
                n_gen_save_path = os.path.join(norm_save_path, f"n_gen_{n_gen}")
                os.makedirs(n_gen_save_path, exist_ok=True)


                # -----------------------
                # Test
                # -----------------------
                overall_test_results_n_gen = {
                    f"E_SC_n{n_gen}":  overall_test_results.get(f"MGC_n{n_gen}", {}),
                    f"L_SC_n{n_gen}":  overall_test_results.get(f"L_MGC_n{n_gen}", {}),
                    f"ML_SC_n{n_gen}": overall_test_results.get(f"ML_MGC_n{n_gen}", {}),
                    f"H_NORM_SC_n{n_gen}":  overall_test_results.get(f"H_MGC_n{n_gen}", {}),
                    f"H_UNNORM_SC_n{n_gen}": overall_test_results.get(f"H_UNNORM_MGC_n{n_gen}", {}),
                    f"B_SC_n{n_gen}":  overall_test_results.get(f"B_MGC_n{n_gen}", {}),
                    f"MEDIAN_SC_n{n_gen}": overall_test_results.get(f"MEDIAN_MGC_n{n_gen}", {}),
                    # f"ARITHMETIC_MEAN_SC_n{n_gen}":   overall_test_results.get(f"MEAN_MGC_n{n_gen}", {}),
                    # f"GEOMETRIC_MEAN_SC_n{n_gen}": overall_test_results.get(f"GEOMETRIC_MEAN_MGC_n{n_gen}", {}),
                    f"MAX_SC_n{n_gen}":    overall_test_results.get(f"MAX_MGC_n{n_gen}", {}),
                    f"Gibbs_0_5_SC_n{n_gen}": overall_test_results.get(f"GIBBS_0_5_MGC_n{n_gen}", {}),
                    f"Gibbs_0_75_SC_n{n_gen}": overall_test_results.get(f"GIBBS_0_75_MGC_n{n_gen}", {}),  
                    f"Gibbs_1_25_SC_n{n_gen}": overall_test_results.get(f"GIBBS_1_25_MGC_n{n_gen}", {}),      
                    f"Cold_0_5_SC_n{n_gen}": overall_test_results.get(f"COLD_0_5_MGC_n{n_gen}", {}),
                    f"Cold_0_75_SC_n{n_gen}": overall_test_results.get(f"COLD_0_75_MGC_n{n_gen}", {}),
                    f"Cold_1_25_SC_n{n_gen}": overall_test_results.get(f"COLD_1_25_MGC_n{n_gen}", {}),
                    # f"Dirichlet_SC_n{n_gen}": overall_test_results.get(f"DIRICHLET_MGC_n{n_gen}", {}),
                    # f"B_Exp_SC_n{n_gen}": overall_test_results.get(f"B_EXP_MGC_n{n_gen}", {}),
                    f"MIN_SC_n{n_gen}": overall_test_results.get(f"MIN_MGC_n{n_gen}", {}),
                    # f"EXP_SC_n{n_gen}": overall_test_results.get(f"EXP_MGC_n{n_gen}", {}),
                }
                
                save_json("test_overall_results", overall_test_results_n_gen, n_gen_save_path)
                
                overall_save_path = os.path.join(n_gen_save_path, "overall_results_by_measure")
                os.makedirs(overall_save_path, exist_ok=True)
                    
                for key, value in overall_test_results_n_gen.items():
                    save_json("test_" + key.lower(), value, overall_save_path)

                # Save individual per‐example results for every strategy
                individual_folder = os.path.join(n_gen_save_path, "individual_results")
                os.makedirs(individual_folder, exist_ok=True)
                for strat, strat_dict in test_results.items():
                    if strat == "overall_results":
                        continue
                    strat_data = strat_dict[norm_type]
                    # strat_data is either a list (e.g. sgc_results) or a dict mapping n_gen→list
                    if isinstance(strat_data, dict):
                        data = strat_data.get(n_gen, [])
                    else:
                        data = strat_data
                    out_path = os.path.join(individual_folder, f"{strat}.json")
                    # with open(out_path, "w") as f:
                    #     json.dump(data, f, indent=4)

            # Save the overall results for the entire splits under this normalization for this run
            # (without subdividing by n_gen)
            overall_results_path = os.path.join(norm_save_path, "overall_results.json")
            with open(overall_results_path, "w") as f:
                json.dump(
                    {
                        "calibration_validation": overall_calib_validation_results,
                        "test": overall_test_results,
                    },
                    f,
                    indent=4,
                )

        # Finalize any additional logging or wandb
        if args.wandb:
            wandb.finish()

        # ------------------------------------------------------------------------------------------
        # Aggregate results after this run
        # ------------------------------------------------------------------------------------------
        for norm_type in ["normalised"]:
            # Pull out the relevant dictionaries
            overall_calib_validation_results = calib_validation_results["overall_results"][norm_type]
            overall_test_results = test_results["overall_results"][norm_type]

            # Accumulate for validation_train
            for key, value in overall_calib_validation_results.items():
                if key not in aggregated_results[norm_type]["calibration_validation"]:
                    aggregated_results[norm_type]["calibration_validation"][key] = {}
                accumulate_metrics(
                    aggregated_results[norm_type]["calibration_validation"][key], value
                )

            # Accumulate for test
            for key, value in overall_test_results.items():
                if key not in aggregated_results[norm_type]["test"]:
                    aggregated_results[norm_type]["test"][key] = {}
                accumulate_metrics(aggregated_results[norm_type]["test"][key], value)

    # ----------------------------------------------------------------------------------------------
    # Save aggregated results across all runs
    # ----------------------------------------------------------------------------------------------
    for norm_type in ["normalised"]:
        aggregated_save_path = os.path.join(
            save_path, f"aggregated_{norm_type}_results.json"
        )
        with open(aggregated_save_path, "w") as f:
            json.dump(aggregated_results[norm_type], f, indent=4)


