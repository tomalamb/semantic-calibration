import torch as t
from torch.nn import functional as F

def ce_with_logit_selective_smoothing_loss(logits, labels, alpha=0.5, ignore_index=-100):
    """
    Cross-Entropy + Selective Logit Smoothing (full-vector) for incorrect predictions.

    Loss = (1/N) * sum_{valid i} CE(logits[i], labels[i])
         + alpha * mean_{i:pred_i != labels_i}( ||logits[i]||_2^2 ),
    where N = # of valid (labels != ignore_index) tokens.
    """
    # 1) Mask out ignored positions
    valid_mask = (labels != ignore_index)
    logits = logits[valid_mask]
    labels = labels[valid_mask]

    # 2) If nothing left, return zero on correct device/dtype
    if logits.numel() == 0:
        return logits.new_tensor(0.)

    # 3) Number of valid tokens as a float tensor
    N = valid_mask.sum().to(dtype=logits.dtype)

    # 4) CE term: sum then average by N
    ce_loss = F.cross_entropy(logits, labels, reduction='sum') / N

    # 5) Find incorrect predictions
    preds = logits.argmax(dim=-1)
    incorrect = preds != labels

    # 6) Smoothing term: mean(||full-vector||^2) over incorrect subset
    if incorrect.any():
        bad = logits[incorrect]              # (M, V)
        sq_norms = bad.pow(2).sum(dim=-1)    # (M,)
        smooth_loss = alpha * sq_norms.mean()
    else:
        smooth_loss = logits.new_tensor(0.)

    return ce_loss + smooth_loss

def ce_with_sentence_logit_selective_smoothing_loss(
    logits: t.Tensor,            # (batch_size, seq_len, vocab_size)
    labels: t.Tensor,            # (batch_size, seq_len)
    is_correct: t.Tensor,        # (batch_size,) 0 or 1
    alpha: float = 0.5,
    ignore_index: int = -100,
    normalise = True
) -> t.Tensor:
    device = logits.device
    
    # --------------------- 1) Cross-Entropy Over Valid Tokens ---------------------
    # (same as before, flatten ignoring ignore_index, then average)
    batch_size, seq_len, vocab_size = logits.shape
    valid_mask = (labels != ignore_index)  # (B, L)
    logits_2d = logits.view(-1, vocab_size) 
    labels_1d = labels.view(-1)
    

    valid_indices = valid_mask.view(-1).nonzero(as_tuple=True)[0]
    valid_logits = logits_2d[valid_indices]  # shape (N_valid, vocab_size)
    valid_labels = labels_1d[valid_indices]  # shape (N_valid,)
    
    if valid_labels.numel() == 0:
        ce_loss = t.tensor(0.0, device=device, requires_grad=True)
    else:
        log_probs = F.log_softmax(valid_logits, dim=-1)
        row_idx = t.arange(len(valid_labels), device=device)
        ce_loss = -log_probs[row_idx, valid_labels].mean()  # average CE

    incorrect_mask = (is_correct == 0)       # shape (B,)

    num_incorrect = incorrect_mask.sum()


    if num_incorrect > 0:

        token_sq_sum = (logits ** 2).sum(dim=-1)  # sum over vocab dimension

        if normalise:
            # We take the mean here 
            seq_sq_sum = token_sq_sum.mean(dim=-1)  # mean over L
        else:
            seq_sq_sum = token_sq_sum.sum(dim=-1)  # sum over L

        incorrect_seq_penalties = seq_sq_sum[incorrect_mask]
        
        seq_penalty_mean = incorrect_seq_penalties.mean()
        
        # scale by alpha
        smoothing_loss = alpha * seq_penalty_mean
    else:
        smoothing_loss = t.tensor(0.0, device=device)

    # --------------------- 3) Combine ---------------------
    total_loss = ce_loss + smoothing_loss
    return total_loss


def ce_with_select_smoothing_loss(logits, labels, alpha=0.5, ignore_index=-100):
    """
    Cross-Entropy Loss with Selective Uniform Smoothing for Incorrect Predictions.

    This loss function always applies the cross-entropy loss to all valid predictions.
    Additionally, for incorrect predictions, it adds a uniform distribution loss
    scaled by alpha.

    Parameters:
    - logits: Tensor of shape (batch_size * seq_length, vocab_size)
    - labels: Tensor of shape (batch_size * seq_length)
    - alpha: Weight for the uniform loss added to incorrect predictions (default 0.5)
    - ignore_index: Label to ignore in the loss computation (default -100)

    Returns:
    - total_loss: Scalar tensor representing the combined loss
    """
    # Ensure logits and labels are on the same device
    device = logits.device

    # 1. Filter out ignored positions
    valid_mask = (labels != ignore_index)  # Shape: (N,)
    logits = logits[valid_mask]            # Shape: (N', C)
    labels = labels[valid_mask]            # Shape: (N',)

    if logits.numel() == 0:
        return t.tensor(0.0, device=device, dtype=logits.dtype)

    # 2. Compute log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (N', C)

    # 3. Compute cross-entropy loss for all valid tokens
    # Gather the log probabilities corresponding to the true labels
    true_log_probs = log_probs[t.arange(logits.size(
        0), device=device), labels]  # Shape: (N',)
    ce_loss = -true_log_probs.sum()  # Scalar

    # 4. Determine predicted classes
    _, predicted_classes = logits.max(dim=-1)  # Shape: (N',)

    # 5. Identify incorrect predictions
    incorrect_mask = (predicted_classes != labels)  # Shape: (N',)
    num_incorrect = incorrect_mask.sum()

    # 6. Compute uniform loss for incorrect predictions
    if num_incorrect > 0:
        # Extract log_probs for incorrect predictions
        incorrect_log_probs = log_probs[incorrect_mask]  # Shape: (N_i, C)
        # Compute the uniform loss: - (1/C) * sum_j log p_{i,j} for each incorrect i
        uniform_loss = -(incorrect_log_probs.sum(dim=-1) /
                         logits.size(-1)).sum()  # Scalar
        # Scale the uniform loss by alpha
        total_uniform_loss = alpha * uniform_loss  # Scalar
    else:
        total_uniform_loss = t.tensor(0.0, device=device, dtype=logits.dtype)

    # 7. Combine cross-entropy loss and uniform loss
    total_loss = ce_loss + total_uniform_loss  # Scalar

    # 8. Normalize by the number of valid tokens
    total_loss = total_loss / valid_mask.sum()

    return total_loss


def adaptive_loss(logits, labels, alpha=0.5, ignore_index=-100):
    """
    Adaptive loss function that combines cross-entropy loss for correct predictions
    and a uniform distribution loss for incorrect predictions.

    logits: Tensor of shape (batch_size * seq_length, vocab_size)
    labels: Tensor of shape (batch_size * seq_length)
    alpha: Weight for the uniform loss (default 0.5)
    ignore_index: Label to ignore in the loss computation (default -100)
    """
    # Filter out ignored positions
    valid_mask = (labels != ignore_index)
    logits = logits[valid_mask]
    labels = labels[valid_mask]

    if logits.numel() == 0:
        return t.tensor(0.0, device=logits.device)

    # Compute log-probabilities for the valid positions
    log_probs = F.log_softmax(logits, dim=-1)

    # Predicted classes (argmax over the vocabulary dimension)
    _, predicted_classes = logits.max(dim=-1)

    # Mask for correct and incorrect predictions
    correct_mask = (predicted_classes == labels)
    incorrect_mask = ~correct_mask

    # Loss for correct predictions: cross-entropy loss with weight (1 - alpha)
    if correct_mask.sum() > 0:
        correct_log_probs = log_probs[correct_mask, labels[correct_mask]]
        correct_loss = -(1 - alpha) * correct_log_probs.sum()
    else:
        correct_loss = t.tensor(0.0, device=logits.device)

    # Loss for incorrect predictions: uniform distribution over all classes
    if incorrect_mask.sum() > 0:
        incorrect_log_probs = log_probs[incorrect_mask]
        # Normalize by the number of classes (vocab size = logits.size(-1))
        uniform_loss = - alpha * \
            (incorrect_log_probs.sum(dim=-1) / logits.size(-1)).sum()
    else:
        uniform_loss = t.tensor(0.0, device=logits.device)

    # Combine the correct loss and the uniform loss
    total_loss = correct_loss + uniform_loss

    # Normalize by the number of valid tokens (i.e., positions that aren't ignore_index)
    total_loss = total_loss / valid_mask.sum()

    return total_loss