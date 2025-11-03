import numpy as np
from sklearn.metrics import roc_auc_score
import re
from rapidfuzz import fuzz
from evaluate import load
from scores.processing.isoreg_impl import isotonic_fit

def compute_ece(ece_values, num_bins=15):
    """
    Compute the Expected Calibration Error (ECE) from scratch.

    Parameters:
    ece_values (dict): Dictionary with keys 'correct' and 'confidence' containing arrays of correctness (0 or 1) and confidence scores.
    num_bins (int): Number of bins to use for calibration calculation.

    Returns:
    float: The computed ECE value.
    """
    correct = np.array(ece_values["correct"])
    confidence = np.array(ece_values["confidence"])

    N = len(correct)
    # Bin edges from 0.0 to 1.0 inclusive
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    # Assign each confidence to a bin
    bin_indices = np.digitize(confidence, bin_edges,
                              right=True) - 1  # Bins are 0-indexed

    # Initialize arrays to store accuracy, confidence, and counts per bin
    acc_bin = np.zeros(num_bins)
    conf_bin = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    # Aggregate data per bin
    for b in range(num_bins):
        in_bin = bin_indices == b
        bin_sizes[b] = np.sum(in_bin)
        if bin_sizes[b] > 0:
            acc_bin[b] = np.mean(correct[in_bin])
            conf_bin[b] = np.mean(confidence[in_bin])

    # Compute the weight of each bin
    bin_weights = bin_sizes / N

    # Compute ECE
    ece = np.sum(bin_weights * np.abs(acc_bin - conf_bin))

    return ece

def compute_ace(ece_values, num_bins=15):
    """
    Compute the Adaptive Calibration Error (ACE).

    Parameters:
        ece_values (dict): Dictionary with keys 'correct' and 'confidence' containing arrays
                           of correctness (0 or 1) and confidence scores.
        num_bins (int): Number of adaptive-equal-mass bins to use for calibration calculation.

    Returns:
        float: The computed ACE value.
    """
    correct = np.array(ece_values["correct"])
    confidence = np.array(ece_values["confidence"])
    N = len(correct)
    
    if N == 0:
        return 0.0  # Handle empty input case
    
    # Sort by confidence
    sorted_indices = np.argsort(confidence)
    sorted_conf = confidence[sorted_indices]
    sorted_corr = correct[sorted_indices]

    # Compute sizes for equal-mass bins
    base_size = N // num_bins
    remainder = N % num_bins
    bin_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_bins)]

    acc_bin = []
    conf_bin = []
    start = 0
    for size in bin_sizes:
        end = start + size
        if size > 0:
            acc_bin.append(sorted_corr[start:end].mean())
            conf_bin.append(sorted_conf[start:end].mean())
        start = end  # Always update start, even if size was zero

    # Compute ACE: mean absolute gap across bins with data
    ace = np.mean(np.abs(np.array(acc_bin) - np.array(conf_bin)))
    return ace


def compute_corp_decomposition(ece_values, bootstraps=250, confidence_level=0.95):
    """
    Compute CORP Brier‐score decomposition.
    Dimitriadis et al. 2021, "Stable reliability diagrams for probabilistic classifiers"

    Parameters:
        ece_values (dict):
            'correct'    → array of 0/1 labels
            'confidence' → array of predicted probabilities
        bootstraps (int):    number of bootstrap resamples for bands (passed to isotonic_fit)
        confidence_level (float): e.g. 0.95 for 95% bands (passed to isotonic_fit)

    Returns:
        tuple:
          MCB  – miscalibration (Sc – Sr)
          DSC  – discrimination (Sr – Ru)
          UNC  – uncertainty baseline (Ru)
          Brier_orig   – original Brier score (Sc)
          Brier_recal  – recalibrated Brier score (Sr)
    """
    # unpack and sort
    y = np.asarray(ece_values["correct"])
    s = np.asarray(ece_values["confidence"])
    order = np.argsort(s)
    s_sorted = s[order]
    y_sorted = y[order]

    # PAV‐based isotonic fit
    iso = isotonic_fit(
        fcst=s_sorted,
        obs=y_sorted,
        functional="mean",
        bootstraps=bootstraps,
        confidence_level=confidence_level
    )
    pav = iso["regression_func"](s_sorted)

    # Brier scores
    Sc = np.mean((y_sorted - s_sorted) ** 2)      # original
    Sr = np.mean((y_sorted - pav) ** 2)           # recalibrated
    Ru = np.mean((y_sorted - np.mean(y_sorted)) ** 2)  # constant‐forecast

    # Decomposition
    MCB = Sc - Sr
    DSC = Ru - Sr
    UNC = Ru
    
    original_brier = Sc
    recalibrated_brier = Sr

    return MCB, DSC, UNC, original_brier, recalibrated_brier


def compute_aurac(uncertainty_values, uncertainty=True):
    """
    Compute the Area Under the Rejection Accuracy Curve (AURAC).

    AURAC is the normalized area under the curve of model accuracy 
    versus retention fraction (i.e. 1 − rejection rate). Equivalently,
    if one sorts all predictions by descending confidence and computes
    cumulative accuracy at each prefix length k = 1,...,n, then

        AURAC = (1/n) * sum_{k=1}^n [ cumulative_accuracy(k) ].

    Parameters
    ----------
    uncertainty_values : dict
        - 'correct': list or array of binary indicators (1=correct, 0=incorrect).
        - 'uncertainty': list or array of uncertainty scores (higher = more uncertain).
        - 'confidence': list or array of confidence scores (higher = more confident).
    uncertainty : bool, default=True
        If True, uses the 'uncertainty' field (inverting it to form confidence).
        If False, uses the 'confidence' field directly.

    Returns
    -------
    float
        AURAC value in [0, 1].
    """

    # 1) Load and validate arrays
    correct = np.asarray(uncertainty_values.get('correct', []), dtype=int)
    if uncertainty:
        vals = np.asarray(uncertainty_values.get('uncertainty', []), dtype=float)
        # invert uncertainty → confidence
        conf = -vals
    else:
        conf = np.asarray(uncertainty_values.get('confidence', []), dtype=float)

    if correct.shape[0] != conf.shape[0]:
        raise ValueError("Length mismatch between 'correct' and 'confidence/uncertainty'.")

    n = correct.shape[0]
    if n == 0:
        raise ValueError("Empty input; cannot compute AURAC.")

    # 2) Sort by descending confidence
    order = np.argsort(-conf)
    sorted_correct = correct[order]

    # 3) Cumulative accuracy at each prefix k
    cum_correct = np.cumsum(sorted_correct)
    ks = np.arange(1, n + 1)
    acc_k = cum_correct / ks

    # 4) AURAC = mean accuracy over all prefixes
    aurac = np.mean(acc_k)
    return aurac

def compute_rejection_accuracy(values, rejection_rate, uncertainty=True):
    """
    Compute the accuracy on the (1 - rejection_rate) most‐confident (or least‐uncertain) examples.
    """
    correct = np.asarray(values["correct"], dtype=int)
    if uncertainty:
        scores = np.asarray(values["uncertainty"], dtype=float)
        # reject most-uncertain first
        order = np.argsort(-scores)
    else:
        scores = np.asarray(values["confidence"], dtype=float)
        # reject least-confident first
        order = np.argsort(scores)

    N = len(scores)
    num_reject = int(np.floor(N * rejection_rate))
    # keep the rest
    accepted_idx = order[num_reject:]
    
    # rejection accuracy = mean(correctness in retained set)
    return correct[accepted_idx].mean()

def compute_auroc(uncertainty_values, uncertainty=True):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUROC) using uncertainty values.

    Parameters:
    uncertainty_values (dict): Dictionary with keys 'correct' and 'uncertainty' or 'confidence'
                               containing lists of correctness (0 or 1) and uncertainty/confidence scores.
    uncertainty (bool): If True, use 'uncertainty' scores; if False, use 'confidence' scores.

    Returns:
    float: The computed AUROC value.
    """

    correct = np.array(uncertainty_values.get("correct", []))

    if uncertainty:
        values = np.array(uncertainty_values.get("uncertainty", []))
        labels = 1 - correct  # Flip labels: 1 indicates incorrect predictions
    else:
        values = np.array(uncertainty_values.get("confidence", []))
        labels = correct       # 1 indicates correct predictions

    # Check if input lengths match
    if len(correct) != len(values):
        raise ValueError(
            "The length of 'correct' and 'uncertainty/confidence' lists must be equal.")

    # Check if labels have both classes (0 and 1)
    if len(np.unique(labels)) < 2:
        # Only one class present in labels; AUROC is not defined
        # Assign AUROC of 0.5, representing random guessing
        auroc = 0.5
    else:
        # AUROC computation using sklearn's roc_auc_score function
        auroc = roc_auc_score(labels, values)

    return auroc

def compute_selective_auroc(values, rejection_rate, uncertainty=False):
    """
    Compute the selective AUROC by rejecting a fraction of examples based on the confidence/uncertainty scores.
    
    Parameters:
        values (dict): A dictionary with keys:
            - "correct": list/array of binary correctness values (1 for correct, 0 for error)
            - "confidence" or "uncertainty": list/array of scores in [0,1]. 
              If uncertainty is False (default), these are treated as confidence scores 
              (with higher meaning more confident). If uncertainty is True, they are treated
              as uncertainty scores (with higher meaning less confident).
        rejection_rate (float): The fraction of examples to reject (between 0 and 1).
        uncertainty (bool): If True, treat the provided scores as uncertainty; 
                            if False (default), treat them as confidence.
    
    Returns:
        float: The AUROC computed on the subset of accepted examples.
    """
    # Get the binary correctness array
    correct = np.array(values.get("correct", []))
    
    if uncertainty:
        # When using uncertainty, higher scores indicate more uncertainty.
        # We want to reject the examples with the highest uncertainty.
        scores = np.array(values.get("uncertainty", []))
        # Sort indices in descending order of uncertainty (most uncertain first)
        sorted_indices = np.argsort(-scores)
    else:
        # When using confidence, higher scores indicate more confidence.
        # We want to reject the examples with the lowest confidence.
        scores = np.array(values.get("confidence", []))
        # Sort indices in ascending order (lowest confidence first)
        sorted_indices = np.argsort(scores)
    
    N = len(scores)
    num_reject = int(N * rejection_rate)
    
    # Determine accepted indices:
    if uncertainty:
        # Reject the top num_reject most uncertain examples.
        accepted_indices = sorted_indices[num_reject:]
        # For uncertainty-based AUROC, we typically flip the labels (i.e. we want high uncertainty for errors)
        labels = 1 - correct[accepted_indices]
    else:
        # Reject the bottom num_reject examples (least confident).
        accepted_indices = sorted_indices[num_reject:]
        labels = correct[accepted_indices]
    
    filtered_scores = scores[accepted_indices]
    
    # If there's only one class present, AUROC is not defined; return 0.5 (random performance)
    if len(np.unique(labels)) < 2:
        return 0.5
    else:
        return roc_auc_score(labels, filtered_scores)

def compute_prr(values, uncertainty=False):
    """
    Compute the Prediction Rejection Ratio (PRR).

    Parameters:
        uncertainties (np.array): Array of uncertainty scores 
                                  (higher means more uncertain).
        correctness (np.array): Binary array where 1 indicates correct prediction,
                                  and 0 indicates error.

    Returns:
        float: The PRR value.
    """
    
    correctness = np.array(values.get("correct", []))
    if uncertainty:
        uncertainties = np.array(values.get("uncertainty", []))
    else:
        # Flip measure.
        uncertainties = - np.array(values.get("confidence", []))
            
    # Total number of examples
    N = len(uncertainties)
    
    # Base error (error rate when no examples are rejected)
    base_error = 1 - np.mean(correctness)
    
    # Number (and fraction) of misclassifications
    misclassifications = np.sum(1 - correctness)
    misclassification_fraction = misclassifications / N

    # Sort examples by uncertainty descending (most uncertain first)
    sorted_indices = np.argsort(-uncertainties)
    sorted_correctness = correctness[sorted_indices]
    
    # Initialize lists to store rejection ratios and corresponding error rates.
    rejection_ratios = []
    error_rates = []
    
    # For k=0 to N, consider rejecting the k most uncertain examples.
    for k in range(0, N + 1):
        # Accepted examples: those not rejected
        accepted = sorted_correctness[k:]
        if len(accepted) > 0:
            error_rate = 1 - np.mean(accepted)
        else:
            error_rate = 0.0
        r = k / N  # rejection ratio
        rejection_ratios.append(r)
        error_rates.append(error_rate)
    
    rejection_ratios = np.array(rejection_ratios)
    error_rates = np.array(error_rates)
    
    # Compute the area under the actual rejection curve (AR_actual)
    AR_actual = np.trapz(error_rates, rejection_ratios)
    
    # Define the "random" rejection curve:
    # When rejecting randomly, the expected error decreases linearly from base_error at r=0 to 0 at r=1.
    random_error_rates = base_error * (1 - rejection_ratios)
    AR_random = np.trapz(random_error_rates, rejection_ratios)
    
    # Define the "oracle" rejection curve:
    # With perfect uncertainty, if you reject exactly the misclassified examples (r = misclassification_fraction),
    # the error drops to 0; before that, the error drops linearly.
    oracle_error_rates = np.maximum(0, base_error * (1 - np.minimum(rejection_ratios / misclassification_fraction, 1)))
    AR_oracle = np.trapz(oracle_error_rates, rejection_ratios)
    
    # The area difference achieved by the uncertainty-based rejection is:
    AR_runs = AR_random - AR_actual
    AR_orc = AR_random - AR_oracle
    
    # Compute PRR (handle the case where AR_orc is zero)
    prr = AR_runs / AR_orc if AR_orc > 0 else np.nan
    return prr

# Check SQuAD F1 score
SQUAD_METRIC = load("squad", keep_in_memory=True)
ROUGE_METRIC = load("rouge", keep_in_memory=True)


def soft_match(pred, refs):
    pred = pred.lower()

    if isinstance(refs, (list, tuple)):
        # Lowercase each reference
        ref_list = [r.lower() for r in refs]

        return 1 if any(
            re.search(r"\b" + re.escape(r) + r"\b", pred)
            for r in ref_list
        ) else 0

    single_ref = refs.lower()
    return 1 if re.search(r"\b" + re.escape(single_ref) + r"\b", pred) else 0


# Define the fuzzy match function with a reasonable threshold
def fuzzy_match(pred, refs, threshold=90):
    """
    Check if the prediction matches the reference(s) with a fuzzy similarity score.

    Args:
        pred (str): The predicted text.
        refs (str or list): A single reference text or a list of reference texts.
        threshold (int): The similarity threshold (default: 90).

    Returns:
        bool: True if a match is found, otherwise False.
    """
    
    # If refs is a single string, wrap it in a list for uniform processing
    if not isinstance(refs, (list, tuple)):
        refs = [refs]

    pred_lower = pred.lower()
    for ref in refs:
        if fuzz.ratio(pred_lower, ref.lower()) >= threshold:
            return True
    return False

ISO_PATTERN = re.compile(r'(\d{4}-\d{2}(?:-\d{2})?)')

def is_iso_date(text: str) -> bool:
    """
    Return True if text contains an ISO date substring:
      - YYYY-MM or
      - YYYY-MM-DD
    """
    return bool(ISO_PATTERN.search(text))

def extract_iso_date(text: str) -> str | None:
    """
    Return the first ISO date substring (YYYY-MM or YYYY-MM-DD) found in text,
    or None if none exists.
    """
    m = ISO_PATTERN.search(text)
    return m.group(1) if m else None


def match_date_reference(single_pred: str, ref_list: list[str]) -> bool:
    """
    Given a prediction string which *contains* an ISO date (but may have
    extra punctuation or words), extract that ISO date and compare it
    against the refs at full-date, month-year, or year-only granularity.
    """
    iso = extract_iso_date(single_pred)
    if not iso:
        return False

    year_month_day = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso)
    if year_month_day:
        year, month = year_month_day.group(1), year_month_day.group(2)
        # a) exact YYYY-MM-DD
        if any(r == iso for r in ref_list if re.fullmatch(r"\d{4}-\d{2}-\d{2}", r)):
            return True
        # b) fallback YYYY-MM
        if any(re.fullmatch(rf"{year}-{month}", r) for r in ref_list):
            return True
        # c) fallback YYYY
        if any(re.fullmatch(rf"{year}", r) for r in ref_list):
            return True
        return False

    year_month = re.fullmatch(r"(\d{4})-(\d{2})", iso)
    if year_month:
        year, _ = year_month.group(1), year_month.group(2)
        # a) exact YYYY-MM
        if any(r == iso for r in ref_list if re.fullmatch(r"\d{4}-\d{2}", r)):
            return True
        # b) fallback YYYY
        if any(re.fullmatch(rf"{year}", r) for r in ref_list):
            return True
        return False

    return False

def evaluate_answer(pred, refs):
    """
    Return 1 if any candidate in `pred` matches `refs` under any of:
      1) whole‐word soft match,
      2) fuzzy ratio ≥ 90,
      3) SQuAD F1 > 50,
      4) ROUGE-L > 0.3.
    Otherwise return 0.

    `pred` may be a single string _or_ a list/tuple of strings.
    `refs` may be a single string or a list of strings (as before).
    """
    ref_list = refs if isinstance(refs, (list, tuple)) else [refs]
    
    # Normalize preds to a list
    if isinstance(pred, (list, tuple)):
        pred_list = pred
    else:
        pred_list = [pred]
        
    base_references = [{
        "id": "1",
        "answers": {"text": ref_list, "answer_start": [0] * len(ref_list)}
    }]

    # Now loop over each candidate in pred_list; return early if any match
    for single_pred in pred_list:
        single_pred = single_pred.strip()
        if not single_pred:
            continue
        
        if is_iso_date(single_pred):
            return 1 if match_date_reference(single_pred, ref_list) else 0

        # 1) Soft (whole‐word) match
        if soft_match(single_pred, refs) == 1:
            return 1

        # 2) Fuzzy match
        if fuzzy_match(single_pred, refs, threshold=90):
            return 1

        # 3) Prepare for SQuAD F1 and ROUGE-L
        predictions = [{"id": "1", "prediction_text": single_pred}]

        # 4) Compute SQuAD F1
        squad_results = SQUAD_METRIC.compute(
            predictions=predictions, references=base_references
        )
        if squad_results.get("f1", 0) > 50.0:
            return 1

        # 5) Compute ROUGE-L
        for single_ref in ref_list:
            rouge_result = ROUGE_METRIC.compute(
                predictions=[single_pred], references=[single_ref]
            )
            if rouge_result.get("rougeL", 0.0) > 0.3:
                return 1
    # If none of the candidates passed any test, return 0
    return 0