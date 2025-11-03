import numpy as np
import torch as t
import torch.nn.functional as F
from typing import Mapping, List, Any
from .text_utils import (
    extract_question_only,
    clean_and_preprocess_text,
    clean_response
)
import random
from .evaluation_metrics import evaluate_answer

def compute_clusters_and_lls(
    generated_texts: list[str],
    prompt: str,
    args,
    tokenizer,
    model,
    device,
    temperature: float,
    temperature_head: bool,
    model_id: str,
    support_examples,
    entailment_model,
    entailment_tokenizer,
    pipe,
    use_modernbert: bool,
    nli_device,
) -> tuple[
    list[set[int]],    # clusters
    int,               # num_clusters
    dict[str, list[str]],     # cluster_examples_dict
    dict[str, t.Tensor],      # cluster_lls
    dict[str, t.Tensor],      # cluster_normalized_lls
]:
    """
    1) Build entailment‐pairs among generated_texts, batch through entailment_model or pipe,
       construct a mutual‐entailment adjacency matrix, and cluster by DFS.
    2) If len(generated_texts) == 1, force clusters = [ {0} ].
    3) For each cluster, build cluster_examples_dict, then compute log‐likelihoods & normalized LLs
       exactly as in your original “Sampling LL part” code, storing them in cluster_lls and cluster_normalized_lls.

    Returns (clusters, num_clusters, cluster_examples_dict, cluster_lls, cluster_normalized_lls).
    """
    # ------------------------------------------------------------------------
    # 1) ENTITLEMENT‐BASED CLUSTERING
    # ------------------------------------------------------------------------
    if len(generated_texts) <= 1:
        clusters = [set([0])]
    else:
        entailment_pairs = []
        entailment_cache = {}            # (i,j) → (qa_1, qa_2)
        entailment_results_cache = {}    # (i,j) → logits tensor
        entailment_results_dict = {}     # (i,j) → predicted_label (int)

        # Build all (i,j) with i != j
        for idx1 in range(len(generated_texts)):
            for idx2 in range(len(generated_texts)):
                if idx1 == idx2:
                    continue
                pair_key = (idx1, idx2)
                if pair_key in entailment_cache:
                    continue

                question_part = extract_question_only(prompt).strip()
                answer_i = clean_and_preprocess_text(generated_texts[idx1])
                answer_j = clean_and_preprocess_text(generated_texts[idx2])

                qa_1 = f"Question: {question_part}\nAnswer: {answer_i}"
                qa_2 = f"Question: {question_part}\nAnswer: {answer_j}"
                
                entailment_cache[pair_key] = (qa_1, qa_2)
                entailment_pairs.append((pair_key, (qa_1, qa_2)))

        batch_size_entail = 80
        for batch_start in range(0, len(entailment_pairs), batch_size_entail):
            batch_end = min(batch_start + batch_size_entail, len(entailment_pairs))
            batch_pairs = entailment_pairs[batch_start:batch_end]
            batch_keys = [p[0] for p in batch_pairs]
            batch_texts = [p[1] for p in batch_pairs]
            batch_1 = [tup[0] for tup in batch_texts]
            batch_2 = [tup[1] for tup in batch_texts]

            new_1, new_2, new_keys = [], [], []
            cached_indices, cached_logits = [], []

            for i, key in enumerate(batch_keys):
                if key in entailment_results_cache:
                    cached_indices.append(i)
                    cached_logits.append(entailment_results_cache[key])
                else:
                    new_1.append(batch_1[i])
                    new_2.append(batch_2[i])
                    new_keys.append(key)

            if new_keys:
                if use_modernbert:
                    batched_inputs = [
                        {"text": p1, "text_pair": p2}
                        for p1, p2 in zip(new_1, new_2)
                    ]
                    raw_outputs = pipe(batched_inputs, return_all_scores=True)
                    for key, scores_list in zip(new_keys, raw_outputs):
                        score_map = {entry["label"]: entry["score"] for entry in scores_list}
                        scores = [
                            score_map["contradiction"],
                            score_map["neutral"],
                            score_map["entailment"],
                        ]
                        logits = t.tensor(scores, device=nli_device)
                        entailment_results_cache[key] = logits
                        entailment_results_dict[key] = int(logits.argmax())
                else:
                    encoded_pairs = entailment_tokenizer(
                        new_1, new_2, padding=True, return_tensors="pt"
                    ).to(nli_device)
                    
                    new_entailment_results = entailment_model(**encoded_pairs).logits
                    predicted_labels = t.argmax(new_entailment_results, dim=1).tolist()
                    for i, key in enumerate(new_keys):
                        logits = new_entailment_results[i]
                        entailment_results_cache[key] = logits
                        entailment_results_dict[key] = predicted_labels[i]

            if cached_logits:
                logits_tensor = t.stack(cached_logits)
                predicted_labels = t.argmax(logits_tensor, dim=1).tolist()
                for idx_in_batch, i in enumerate(cached_indices):
                    key = batch_keys[i]
                    entailment_results_dict[key] = predicted_labels[idx_in_batch]

        N = len(generated_texts)
        entailment_matrix = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                label_ij = entailment_results_dict.get((i, j))
                label_ji = entailment_results_dict.get((j, i))
                if label_ij == 2 and label_ji == 2:
                    entailment_matrix[i, j] = 1

        clusters = []
        visited = set()
        for i in range(N):
            if i in visited:
                continue
            stack = [i]
            visited.add(i)
            this_cluster = {i}
            while stack:
                cur = stack.pop()
                for neighbor in range(N):
                    if entailment_matrix[cur, neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        this_cluster.add(neighbor)
            clusters.append(this_cluster)

    # Force single‐text fallback
    clusters = [set([0])] if len(generated_texts) == 1 else clusters
    num_clusters = len(clusters)

    # ------------------------------------------------------------------------
    # 2) SAMPLING LL PART
    # ------------------------------------------------------------------------
    cluster_examples_dict = {}
    cluster_lls = {}
    cluster_normalized_lls = {}

    for cluster_idx, cluster in enumerate(clusters):
        current_cluster_key = f"cluster_{cluster_idx}"
        cluster_examples = [generated_texts[idx] for idx in cluster]
        cluster_examples_dict[current_cluster_key] = cluster_examples

        messages = []
        for output in cluster_examples:
            question_text = extract_question_only(prompt).strip()
            message = [
                {"role": "user", "content": f"Question: {question_text}" if args.few_shot else prompt},
                {"role": "assistant", "content": output}
            ]
            messages.append(
                tokenizer.apply_chat_template(
                    message if support_examples is None else support_examples + message,
                    tokenize=False,
                    add_generation_prompt=False
                )
            )
        
        combined_tokenized_outputs = tokenizer(
            messages,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        if "falcon" in model_id:
            answer_strings = [
                " " + c + tokenizer.eos_token for c in cluster_examples
            ]
        else:
            answer_strings = [
                c + tokenizer.eos_token for c in cluster_examples
            ]

        output_lengths = t.tensor([
            len(tokenizer.tokenize(answer, add_special_tokens=False))
            for answer in answer_strings
        ]).to(device)

        labels = combined_tokenized_outputs.input_ids.clone()
        for i, length in enumerate(output_lengths):
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
            ).logits

        if not temperature_head:
            scaled_logits = outputs / temperature
        else:
            scaled_logits = outputs

        shift_logits = scaled_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        batch_size_ll, seq_length = shift_labels.size()
        batch_indices_ll = (
            t.arange(batch_size_ll, device=device)
            .unsqueeze(1)
            .expand(-1, seq_length)
        )
        seq_indices_ll = (
            t.arange(seq_length, device=device)
            .unsqueeze(0)
            .expand(batch_size_ll, -1)
        )

        log_probs = log_probs[batch_indices_ll, seq_indices_ll, shift_labels]
        valid_mask = (shift_labels != -100).float()
        masked_log_probs = log_probs * valid_mask

        log_likelihoods = masked_log_probs.sum(dim=-1)
        num_valid_tokens = valid_mask.sum(dim=-1)
        num_valid_tokens = t.where(
            num_valid_tokens == 0,
            t.tensor(1.0, device=device),
            num_valid_tokens,
        )
        normalized_log_likelihoods = log_likelihoods / num_valid_tokens

        cluster_lls[current_cluster_key] = log_likelihoods
        cluster_normalized_lls[current_cluster_key] = normalized_log_likelihoods
        
    # Finally clean the cluste_examples_dict for later evalutaion
    for key in cluster_examples_dict:
        cluster_examples_dict[key] = [
            clean_and_preprocess_text(clean_response(txt, args.dset)) for txt in cluster_examples_dict[key]
        ]

    return clusters, num_clusters, cluster_examples_dict, cluster_lls, cluster_normalized_lls


def evaluate_one_cluster_method(
    tag: str,
    cluster_scores: t.Tensor,
    cluster_lls_map: Mapping[str, t.Tensor],
    cluster_examples_dict: Mapping[str, List[str]],
    refs: List[List[str]],
    classes: List[Any],
    y: int,
    prompt: str,
    beam_response: str,
    norm_type: str,
    n_gen: int,
    num_clusters: int,
    metrics: dict,
    args_dset: str,
    eps: float = 1e-12,
):
    """
    Abstracts the common pattern for ML-MGC, H-SC, MGC, L-MGC, B-MGC,
    Median-SC, Mean-SC, Max-SC.  It:
      1. Picks the cluster c* = argmax cluster_scores.
      2. Within c*, picks the single answer with largest log-likelihood ℓ*.
      3. Computes correctness across all members of c*.
      4. Updates metrics[...] under keys that involve 'tag' (e.g. "ml_mgc", "h_mgc", etc.).
      5. Appends one result-dict into metrics[norm_type][f"{tag}_results"][n_gen].
    """
    device = cluster_scores.device
    # 1) Select best cluster index c*
    #    cluster_scores is assumed to be a 1-D tensor of length = num_clusters.
    confidence_c, best_cluster_idx = t.max(cluster_scores, dim=0)
    c_key = f"cluster_{best_cluster_idx.item()}"
    examples_in_c = cluster_examples_dict[c_key]          # List[str]
    lls_in_c      = cluster_lls_map[c_key]                # 1-D tensor of per-example ℓ’s

    # 2) Find the single best ℓ* in this cluster
    best_response_ll, best_example_idx = t.max(lls_in_c, dim=-1)
    chosen_text = examples_in_c[best_example_idx.item()]

    # 3) “Cluster-wide correctness” = 1 if ANY member of this cluster matches gold
    #    First, clean & preprocess all strings in examples_in_c:
    others = [ex for ex in examples_in_c if ex != chosen_text]
    n_more = min(1, len(others))
    if n_more > 0:
        sampled = random.sample(others, k=n_more)
        candidates = [chosen_text] + sampled
    else:
        candidates = [chosen_text]

    gold_list = refs[y]
    correctness = int(evaluate_answer(candidates, gold_list))

    # 4) Update scalar metrics: NLL, NLC, accuracy, Brier, ECE/PRR/AUROC, entropy…
    #    a) Negative log‐likelihood = -ℓ*
    metrics[norm_type][f"total_nll_{tag}"][n_gen] -= best_response_ll.item()
    #    b) Negative log‐confidence = -log(cluster_scores[c*])
    metrics[norm_type][f"total_nlc_{tag}"][n_gen] -= t.log(confidence_c + eps).item()
    #    c) Accuracy
    metrics[norm_type][f"accuracy_{tag}"][n_gen] += correctness

    #    d) ECE/PRR/AUROC for “cluster‐level” confidence = confidence_c
    for metric_name in ["ece", "prr", "auroc"]:
        metrics[norm_type][f"{metric_name}_values_{tag}_conf"][n_gen]["correct"].append(correctness)
        metrics[norm_type][f"{metric_name}_values_{tag}_conf"][n_gen]["confidence"].append(confidence_c.item())
    #    e) ECE/PRR/AUROC for “LNLL‐level” confidence = exp(ℓ*)
    lnll_conf = best_response_ll.exp().item()
    for metric_name in ["ece", "prr", "auroc"]:
        metrics[norm_type][f"{metric_name}_values_{tag}_lnll"][n_gen]["correct"].append(correctness)
        metrics[norm_type][f"{metric_name}_values_{tag}_lnll"][n_gen]["confidence"].append(lnll_conf)

    #    f) Brier scores
    metrics[norm_type][f"total_brier_{tag}_conf"][n_gen] += (confidence_c.item() - correctness) ** 2
    metrics[norm_type][f"total_brier_{tag}_lnll"][n_gen] += (lnll_conf - correctness) ** 2

    #    g) Shannon‐RB entropy  = -∑_c [ p_c * log(p_c ) ]
    rb_entropy = -t.sum(cluster_scores * t.log(cluster_scores + eps))
    metrics[norm_type][f"{tag}_rb_entropy_values"][n_gen]["correct"].append(correctness)
    metrics[norm_type][f"{tag}_rb_entropy_values"][n_gen]["uncertainty"].append(rb_entropy.item())

    #    h) Mean‐cluster (MC) entropy = -mean( log p_c )
    mc_entropy = -t.mean(t.log(cluster_scores + eps))
    metrics[norm_type][f"{tag}_mc_entropy_values"][n_gen]["correct"].append(correctness)
    metrics[norm_type][f"{tag}_mc_entropy_values"][n_gen]["uncertainty"].append(mc_entropy.item())
    
    
    # Compuate the semantic etropy as done in SE paper with beam generation used for correctness. 
    beam_correctness = int(evaluate_answer(beam_response, gold_list)) 
    
    # Compute new entropy as done in SE paper for comparison
    metrics[norm_type][f"{tag}_rb_entropy_values_SE"][n_gen]["correct"].append(beam_correctness)
    metrics[norm_type][f"{tag}_rb_entropy_values_SE"][n_gen]["uncertainty"].append(rb_entropy.item())

    metrics[norm_type][f"{tag}_mc_entropy_values_SE"][n_gen]["correct"].append(beam_correctness)
    metrics[norm_type][f"{tag}_mc_entropy_values_SE"][n_gen]["uncertainty"].append(mc_entropy.item())

    # 5) Push one result‐dict into metrics[norm_type][f"{tag}_results"][n_gen]
    result_dict = {
        "query_text_prompt": prompt,
        "generated_text": chosen_text,
        "actual_label": classes[y],
        "correctness": correctness,
        "confidence": confidence_c.item(),
        "lnll": best_response_ll.item(),
        "num_clusters": num_clusters,
        "cluster_distribution": cluster_scores.tolist(),
        "cluster_examples": cluster_examples_dict,
    }
    metrics[norm_type][f"{tag}_results"][n_gen].append(result_dict)