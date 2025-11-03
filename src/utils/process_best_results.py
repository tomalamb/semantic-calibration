#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate per-run metric JSONs, average where requested, and
report mean / variance / standard-error for each metric. Also
select the best tempera    A("\\caption{Best temperature settings for Temperature Scaling (TS) method across models and datasets. " +
      "Temperatures represent the final optimized values after hyperparameter search.}")re according to calibration metrics.
"""

import itertools
import json
import math
import os
import re

# ──────────────────────────  Configuration  ────────────────────────── #

# Metric used for selecting best hyperparameters on calibration validation set
# Options: "Brier_conf", "ECE_conf", "ACE_conf", "NLL", etc.
SELECTION_METRIC = "Brier_conf"

# Global dictionary to track best temperature settings for TS method
TEMPERATURE_SETTINGS = {}

# ──────────────────────────  I/O helpers  ────────────────────────── #

def extract_temperature_from_run_id(run_id):
    """Extract final optimized temperature value from nested run_id path structure."""
    import re
    
    # Handle nested folder structure like:
    # "initial_temp_1.0_temp_lr_0.0001_weight_decay_0.0_loss_weight_0.5/temp_1.7099681926412034/runs"
    # We want to extract 1.7099681926412034 (the final optimized temperature)
    
    # Look for the final temperature in the nested path structure
    # Pattern: temp_<final_temperature> in the path
    temp_folder_pattern = r'temp_(\d+\.?\d*)'
    matches = re.findall(temp_folder_pattern, run_id)
    
    if matches:
        # Take the last match (most nested) which should be the final temperature
        return float(matches[-1])
    
    # Fallback: look for other temperature patterns
    patterns = [
        r'temperature_(\d+\.?\d*)',
        r't_(\d+\.?\d*)',
        r'(\d+\.?\d*)_temp'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, run_id, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return None

def save_temperature_settings_table():
    """Generate and save LaTeX table showing best temperature settings for Temperature Scaling method."""
    if not TEMPERATURE_SETTINGS:
        print("No temperature settings recorded.")
        return
    
    # Get all models and datasets
    models = list(TEMPERATURE_SETTINGS.keys())
    datasets = set()
    for model_data in TEMPERATURE_SETTINGS.values():
        datasets.update(model_data.keys())
    datasets = sorted(list(datasets))
    
    # Reorder datasets: trivia_qa, natural_questions, squad
    datasets = [d for d in ["trivia_qa", "natural_questions", "squad"] if d in datasets]
    
    # Use the same measures as in best_table_plots.py
    plotted_measures = ["MGC_n10", "L_MGC_n10", "ML_MGC_n10", "B_MGC_n10", "H_UNNORM_MGC_n10", "COLD_MGC_n10", "GIBBS_MGC_n10"]
    
    # Function to find best temperature for collapsed methods
    def find_best_temperature_for_method(model, dataset, method):
        temp_info = TEMPERATURE_SETTINGS.get(model, {}).get(dataset, {}).get(method, {})
        
        if temp_info.get('temperature') is not None:
            return temp_info
        
        # For COLD_MGC_n10 and GIBBS_MGC_n10, find the best variant
        if method in ["COLD_MGC_n10", "GIBBS_MGC_n10"]:
            prefix = method.replace("_MGC_n10", "")  # COLD or GIBBS
            best_score = float('inf')
            best_temp_info = None
            
            # Look for all variants of this method
            model_data = TEMPERATURE_SETTINGS.get(model, {}).get(dataset, {})
            for variant_name, variant_info in model_data.items():
                if variant_name.startswith(prefix) and variant_info.get('temperature') is not None:
                    score = variant_info.get('metric_score', float('inf'))
                    if score < best_score:
                        best_score = score
                        best_temp_info = variant_info
            
            return best_temp_info or temp_info
        
        return temp_info
    
    # Create friendly names mapping (same as best_table_plots.py)
    friendly_names = {"MGC_n10":"E-SC", "L_MGC_n10":"L-SC", "ML_MGC_n10":"ML-SC", "B_MGC_n10":"B-SC", 
                      "H_UNNORM_MGC_n10":"IC-SC", "COLD_MGC_n10":"T-SC", "GIBBS_MGC_n10": "G-SC"}
    
    # Fix collapsed methods that have null temperatures by finding their best variants
    for model in models:
        if model not in TEMPERATURE_SETTINGS:
            continue
        for dataset in datasets:
            if dataset not in TEMPERATURE_SETTINGS[model]:
                continue
            
            # Fix COLD_MGC_n10 and GIBBS_MGC_n10 that have null temperatures
            for measure in ["COLD_MGC_n10", "GIBBS_MGC_n10"]:
                if (measure in TEMPERATURE_SETTINGS[model][dataset] and 
                    TEMPERATURE_SETTINGS[model][dataset][measure].get('temperature') is None):
                    
                    # Find the best variant by looking at all related measures
                    base_name = measure.split("_MGC_n10")[0]  # "COLD" or "GIBBS"
                    best_temp = None
                    best_score = float('inf')
                    best_variant_data = None
                    
                    # Look for variants like COLD_0_5_MGC_n10, GIBBS_0_75_MGC_n10, etc.
                    for variant_name, variant_data in TEMPERATURE_SETTINGS[model][dataset].items():
                        if (variant_name.startswith(base_name + "_") and 
                            "_MGC_n10" in variant_name and 
                            variant_data.get('temperature') is not None):
                            
                            score = variant_data.get('metric_score', float('inf'))
                            if score < best_score:
                                best_score = score
                                best_temp = variant_data['temperature']
                                best_variant_data = variant_data.copy()
                    
                    # Update the collapsed method with the best variant's temperature
                    if best_temp is not None:
                        TEMPERATURE_SETTINGS[model][dataset][measure]['temperature'] = best_temp
                        if best_variant_data:
                            TEMPERATURE_SETTINGS[model][dataset][measure].update(best_variant_data)
    
    tex = []
    A = tex.append
    
    A("\\begin{table}[h!]")
    A("\\footnotesize")
    A("\\centering")
    A("\\setlength{\\tabcolsep}{4pt}")
    A("\\resizebox{\\textwidth}{!}{%")
    
    # Create column alignment
    num_measures = len(plotted_measures)
    align = "ll" + "c" * num_measures
    A("\\begin{tabular}{" + align + "}")
    A("\\toprule")
    
    # Header with confidence measures
    A("Dataset & Model & " + " & ".join([friendly_names.get(m, m) for m in plotted_measures]) + " \\\\")
    A("\\midrule")
    
    # Data rows
    for i, dataset in enumerate(datasets):
        dataset_display = dataset.replace('_', ' ').title()
        
        for j, model in enumerate(models):
            # Simplify model names
            if "Llama" in model:
                model_display = "Llama"
            elif "Qwen" in model:
                model_display = "Qwen"
            elif "Ministral" in model or "Mistral" in model:
                model_display = "Mistral"
            else:
                model_display = model.replace('-', '\\-')  # Fallback with escaped hyphens
            
            # Dataset label (multirow for first model in each dataset)
            if j == 0:
                left = f"\\multirow{{{len(models)}}}{{*}}{{{dataset_display}}}"
            else:
                left = ""
            
            line = [left, model_display]
            
            # Temperature values for each measure
            for measure in plotted_measures:
                temp_info = find_best_temperature_for_method(model, dataset, measure)
                
                if temp_info and 'temperature' in temp_info and temp_info['temperature'] is not None:
                    temp = temp_info['temperature']
                    # Format to exactly 3 significant figures
                    import math
                    if temp == 0:
                        temp_str = "0.00"
                    else:
                        # Calculate number of digits before decimal
                        magnitude = int(math.floor(math.log10(abs(temp))))
                        # Format to 3 significant figures
                        decimal_places = max(0, 2 - magnitude)
                        temp_str = f"{temp:.{decimal_places}f}"
                    
                    line.append(temp_str)
                else:
                    line.append("-")
            
            A(" & ".join(line) + " \\\\")
        
        # Add separator between datasets except for the last one
        if i < len(datasets) - 1:
            A("\\midrule")
    
    A("\\bottomrule")
    A("\\end{tabular}}%")
    A("\\caption{Best temperature settings for Temperature Scaling (TS) method across models and datasets. " +
      "Temperatures represent the final optimized values after hyperparameter search.}")
    A("\\label{tab:temperature_settings}")
    A("\\end{table}")
    
    # Save table
    os.makedirs("results", exist_ok=True)
    out_path = "results/latex_table_temperature_settings.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(tex))
    
    print(f"✓ Temperature settings table saved to: {out_path}")
    
    # Also save the raw data as JSON for reference
    raw_data_path = "results/temperature_settings_data.json"
    with open(raw_data_path, "w") as f:
        json.dump(TEMPERATURE_SETTINGS, f, indent=4)
    print(f"✓ Raw temperature data saved to: {raw_data_path}")


def save_overall_results(results: dict, output_path: str) -> None:
    """Pretty-print a JSON dict to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


# ───────────────────────  basic statistics  ─────────────────────── #

def compute_mean_var_se(values):
    """Return (mean, unbiased variance, SE) from a list of floats or nested lists."""
    # Handle nested lists by flattening
    if isinstance(values, list) and len(values) > 0:
        if isinstance(values[0], list):
            # Flatten nested list
            flat_values = [item for sublist in values for item in sublist]
        else:
            flat_values = values
    else:
        flat_values = values
    
    n = len(flat_values)
    if n == 0:
        return float("inf"), None, None
    
    # Ensure all values are numeric
    numeric_values = []
    for val in flat_values:
        if isinstance(val, (int, float)):
            numeric_values.append(val)
    
    if not numeric_values:
        return float("inf"), None, None
        
    n = len(numeric_values)
    mean = sum(numeric_values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in numeric_values) / (n - 1)   # Bessel-corrected
        se  = math.sqrt(var) / math.sqrt(n)
    else:
        var = se = 0.0
    return mean, var, se


# ────────────────  min / max criterion per submetric  ────────────── #

SUB_METRIC_CRITERIA = {
    "ACC": "max",
    "NLL": "min",
    "ECE_lnll": "min",
    "ECE_conf": "min",
    "ACE_conf": "min",
    "Brier_lnll": "min",
    "Brier_conf": "min",
    "AUROC_lnll": "max",
    "AUROC_conf": "max",
    "semantic_rb_entropy": "max",
    "semantic_mc_entropy": "max",
    "PRR_conf": "max",
    "PRR_lnll": "max",
    "SEL_AUROC_1_conf": "max",
    "SEL_AUROC_1_lnll": "max",
    "SEL_AUROC_5_conf": "max",
    "SEL_AUROC_5_lnll": "max",
    "SEL_AUROC_10_conf": "max",
    "SEL_AUROC_10_lnll": "max",
    "SEL_AUROC_20_conf": "max",
    "SEL_AUROC_20_lnll": "max",
    "SEL_AUROC_30_conf": "max",
    "SEL_AUROC_30_lnll": "max",
}

# ──────────────────  pick best temperature helper  ───────────────── #

def compare_and_update_best(
    best_entry, calib_mean, calib_var, calib_se,
    test_mean, test_var, test_se, temp_value, submetric_name
):
    """Update best_entry if current calibration mean is better."""
    if submetric_name != SELECTION_METRIC:
        return
   
    criterion   = SUB_METRIC_CRITERIA.get(submetric_name, "min")
    current_best = best_entry["calibration"]["mean"]
    current_temp = best_entry["temp"]

    is_better = (
        calib_mean < current_best if criterion == "min" else calib_mean > current_best
    )
    if is_better or current_temp is None:
        best_entry["calibration"] = {
            "mean":           calib_mean,
            "variance":       calib_var,
            "standard_error": calib_se,
        }
        best_entry["test"] = {
            "mean":           test_mean,
            "variance":       test_var,
            "standard_error": test_se,
        }
        best_entry["temp"] = temp_value


# ────────────────────  merge *nX sample-size* keys  ───────────────── #

def elementwise_average(list_of_lists):
    if not list_of_lists:
        return []
    L = len(list_of_lists[0])
    if any(len(arr) != L for arr in list_of_lists):
        raise ValueError("Mismatch in run-array lengths.")
    return [sum(arr[i] for arr in list_of_lists) / len(list_of_lists) for i in range(L)]


def merge_sample_size_variants(variations_dict):
    bucket = {}
    for _name, submetrics in variations_dict.items():
        for metric, arr in submetrics.items():
            if isinstance(arr, list) and all(isinstance(x, (int, float)) for x in arr):
                bucket.setdefault(metric, []).append(arr)
    return {m: elementwise_average(arrs) for m, arrs in bucket.items()}


def add_averaged_metrics_to_block(block):
    """Add *_AVERAGED keys alongside existing metrics."""
    grouped, passthrough = {}, {}
    for name, sub in block.items():
        m = re.match(r"^(.*)_n\d+$", name)
        if m:
            grouped.setdefault(m.group(1), {})[name] = sub
        else:
            passthrough[name] = sub

    new_metrics = {
        f"{core}_AVERAGED": merge_sample_size_variants(variants)
        for core, variants in grouped.items()
    }
    out = passthrough.copy()
    out.update(new_metrics)
    return out


def inject_averaged_metrics(data):
    if "test" in data:
        data["test"] = add_averaged_metrics_to_block(data["test"])
    if "calibration_validation" in data:
        data["calibration_validation"] = add_averaged_metrics_to_block(
            data["calibration_validation"]
        )


def drop_raw_nx_keep_avg(block):
    out = {}
    for name, sub in block.items():
        if re.match(r".*_AVERAGED$", name):
            out[re.sub(r"_AVERAGED$", "", name)] = sub
        elif not re.match(r".*_n\d+$", name):
            out[name] = sub
    return out


# ───────────  recurse through JSON and update best table  ─────────── #

def update_best_results_from_file(calib_dict, test_dict, best, prefix=None, temp=None):
    prefix = prefix or []
    # descend dictionaries
    if isinstance(calib_dict, dict):
        for k, v in calib_dict.items():
            update_best_results_from_file(
                v, test_dict.get(k, {}), best, prefix + [k], temp
            )
        return

    # reached a numeric list
    if not (isinstance(calib_dict, list) and all(isinstance(x, (int, float)) for x in calib_dict)):
        return

    method = prefix[0]
    metric = "__".join(prefix[1:]) if len(prefix) > 1 else "unknown"

    calib_mean, calib_var, calib_se = compute_mean_var_se(calib_dict)
    if isinstance(test_dict, list) and all(isinstance(x, (int, float)) for x in test_dict):
        test_mean, test_var, test_se = compute_mean_var_se(test_dict)
    else:
        test_mean = test_var = test_se = None

    best.setdefault(method, {}).setdefault(
        metric,
        {
            "calibration": {"mean": float("inf"), "variance": None, "standard_error": None}
            if SUB_METRIC_CRITERIA.get(metric, "min") == "min"
            else {"mean": float("-inf"), "variance": None, "standard_error": None},
            "test": {"mean": None, "variance": None, "standard_error": None},
            "temp": None,
        },
    )
    compare_and_update_best(
        best[method][metric],
        calib_mean, calib_var, calib_se,
        test_mean,  test_var,  test_se,
        temp,
        metric,
    )


# ────────────────  walk directory tree and consolidate  ───────────── #

def process_files_recursive(base_folder, metric_type="normalised", model_name=None, dataset_name=None):
    """
    For each confidence measure independently, find the hyperparameter configuration
    that gives the best validation score (based on SELECTION_METRIC), then use the corresponding test results.
    This treats each confidence measure as independent and selects the best hyperparameter
    configuration (including temperature settings) for that specific measure.
    """
    target = f"aggregated_{metric_type}_results.json"

    # 1) collect every run under base_folder (each subfolder represents a hyperparameter config)
    runs = []
    for root, _dirs, files in os.walk(base_folder):
        if target not in files:
            continue
        run_id = os.path.relpath(root, base_folder)
            
        with open(os.path.join(root, target)) as f:
            runs.append((run_id, json.load(f)))

    if not runs:
        return {"calibration_validation": {}, "test": {}}

    # 2) Find all confidence measures present in the data
    all_conf_measures = set()
    for run_id, data in runs:
        cv = data.get("calibration_validation", {})
        all_conf_measures.update(cv.keys())

    final_json = {"calibration_validation": {}, "test": {}}
    
    # Track temperature settings for scalar method (Temperature Scaling)
    is_scalar_method = "scalar" in base_folder
    if is_scalar_method and model_name and dataset_name:
        if model_name not in TEMPERATURE_SETTINGS:
            TEMPERATURE_SETTINGS[model_name] = {}
        if dataset_name not in TEMPERATURE_SETTINGS[model_name]:
            TEMPERATURE_SETTINGS[model_name][dataset_name] = {}
    
    # 3) For each confidence measure independently, find the best hyperparameter configuration
    for conf_measure in all_conf_measures:
        # Initialize best value based on whether we're minimizing or maximizing
        criterion = SUB_METRIC_CRITERIA.get(SELECTION_METRIC, "min")
        best_brier = float('inf') if criterion == "min" else float('-inf')
        best_run_data = None
        best_run_id = None
        
        # Find the run with the best SELECTION_METRIC for this specific confidence measure
        for run_id, data in runs:
            cv = data.get("calibration_validation", {})
            if conf_measure in cv:
                metric_scores = cv[conf_measure].get(SELECTION_METRIC, [])
                if metric_scores:
                    mean_metric = sum(metric_scores) / len(metric_scores)
                    # Check if we want to minimize or maximize this metric
                    criterion = SUB_METRIC_CRITERIA.get(SELECTION_METRIC, "min")
                    is_better = (mean_metric < best_brier if criterion == "min" else mean_metric > best_brier)
                    if is_better:
                        best_brier = mean_metric
                        best_run_data = data
                        best_run_id = run_id
        
        # Store temperature setting for scalar method
        if is_scalar_method and model_name and dataset_name and best_run_id:
            # Extract temperature from run_id (format might be like "temp_1.5" or similar)
            temp_value = extract_temperature_from_run_id(best_run_id)
            TEMPERATURE_SETTINGS[model_name][dataset_name][conf_measure] = {
                'temperature': temp_value,
                'metric_score': best_brier,
                'run_id': best_run_id
            }
        
        # Process the best configuration for this confidence measure
        if best_run_data is not None:
            for section in ["calibration_validation", "test"]:
                section_data = best_run_data.get(section, {})
                if conf_measure in section_data:
                    processed_metrics = {}
                    for metric, values in section_data[conf_measure].items():
                        if isinstance(values, list) and values:
                            mean_val, var_val, se_val = compute_mean_var_se(values)
                            processed_metrics[metric] = {
                                "mean": mean_val,
                                "variance": var_val,
                                "standard_error": se_val
                            }
                    if processed_metrics:
                        final_json[section][conf_measure] = processed_metrics
            
            print(f"      {conf_measure}: best hyperparameters from {best_run_id} ({SELECTION_METRIC}: {best_brier:.4f})")

    # Collapse temperature variants for GIBBS and COLD measures that have multiple settings
    final_json = collapse_temperature_variants(final_json)

    return final_json


def collapse_temperature_variants(data):
    """
    Collapse temperature variants for Gibbs and Cold measures that have multiple settings.
    For example: GIBBS_0_5_MGC_n10, GIBBS_1_25_MGC_n10 -> GIBBS_MGC_n10
    Choose the best variant based on calibration validation Brier_conf score.
    """
    if not isinstance(data, dict):
        return data
        
    collapsed_data = {"calibration_validation": {}, "test": {}}
    
    # First pass: group temperature variants and find the best one based on calibration data
    section_data = data.get("calibration_validation", {})
    base_groups = {}
    passthrough = {}
    best_variants = {}  # Store which variant was chosen for each base group
    
    for conf_measure, metrics in section_data.items():
        # Check if this is a temperature variant of GIBBS or COLD
        # GIBBS variants: GIBBS_0_5_MGC_n10, GIBBS_0_75_MGC_n10, etc. -> GIBBS_MGC_n10
        # COLD variants: COLD_0_5_MGC_n10, COLD_0_75_MGC_n10, etc. -> COLD_MGC_n10
        if conf_measure.startswith("GIBBS_") and "_MGC_n10" in conf_measure:
            base_name = "GIBBS_MGC_n10"
            if base_name not in base_groups:
                base_groups[base_name] = {}
            base_groups[base_name][conf_measure] = metrics
        elif conf_measure.startswith("COLD_") and "_MGC_n10" in conf_measure:
            base_name = "COLD_MGC_n10"
            if base_name not in base_groups:
                base_groups[base_name] = {}
            base_groups[base_name][conf_measure] = metrics
        else:
            # Keep non-temperature variants as-is
            passthrough[conf_measure] = metrics
    
    # Select the best variant for each base group based on calibration validation SELECTION_METRIC
    for base_name, variants in base_groups.items():
        best_variant = None
        criterion = SUB_METRIC_CRITERIA.get(SELECTION_METRIC, "min")
        best_brier = float('inf') if criterion == "min" else float('-inf')
        
        for variant_name, metrics in variants.items():
            metric_data = metrics.get(SELECTION_METRIC, {})
            if isinstance(metric_data, dict) and "mean" in metric_data:
                metric_mean = metric_data["mean"]
                is_better = (metric_mean < best_brier if criterion == "min" else metric_mean > best_brier)
                if is_better:
                    best_brier = metric_mean
                    best_variant = variant_name
        
        if best_variant:
            best_variants[base_name] = best_variant
            collapsed_data["calibration_validation"][base_name] = variants[best_variant].copy()
            print(f"        {base_name}: chose {best_variant} ({SELECTION_METRIC}: {best_brier:.4f})")
    
    # Add the passthrough measures (non-temperature variants) to calibration_validation
    collapsed_data["calibration_validation"].update(passthrough)
    
    # Second pass: apply the same choice to test data
    test_section_data = data.get("test", {})
    test_passthrough = {}
    
    for conf_measure, metrics in test_section_data.items():
        # Check if this is a temperature variant that we collapsed
        found_base = None
        for base_name, chosen_variant in best_variants.items():
            if conf_measure == chosen_variant:
                found_base = base_name
                break
        
        if found_base:
            # This is the chosen variant for a collapsed base group
            collapsed_data["test"][found_base] = metrics.copy()
        elif not (conf_measure.startswith("GIBBS_") and "_MGC_n10" in conf_measure) and not (conf_measure.startswith("COLD_") and "_MGC_n10" in conf_measure):
            # This is not a temperature variant, keep as-is
            test_passthrough[conf_measure] = metrics
    
    # Add the passthrough measures to test data
    collapsed_data["test"].update(test_passthrough)
    
    return collapsed_data


# ─────────────────  Simple cross-loss comparison that actually works  ────────────────── #

def simple_cross_loss_comparison(model_dset_path, base_method_name, confidence_measures, model_name=None, dataset_name=None):
    """
    Collapse the best results based on NLL and SS (adaptive) losses for each method.
    For each confidence measure independently, pick the loss variant with the best 
    calibration validation Brier score. This treats the loss type as another hyperparameter.
    """
    loss_variants = {
        "cross_entropy": f"cross_entropy_{base_method_name}",
        "adaptive": f"adaptive_{base_method_name}"
    }
    
    final_result = {"calibration_validation": {}, "test": {}}
    
    # Load data from both loss variants and extract temperature information
    loss_data = {}
    loss_temp_info = {}  # Store temperature information for each loss variant
    
    for loss_name, loss_folder in loss_variants.items():
        path = os.path.join(model_dset_path, loss_folder, "final_overall_results_normalised.json")
        if os.path.exists(path):
            with open(path) as f:
                loss_data[loss_name] = json.load(f)
            
            # Extract temperature information by looking at the subfolder structure
            loss_temp_info[loss_name] = extract_temperature_info_from_loss_folder(
                os.path.join(model_dset_path, loss_folder)
            )
    
    if not loss_data:
        return final_result
    
    print(f"    Comparing {list(loss_data.keys())} for {base_method_name}")
    
    # Track temperature settings for scalar method
    is_scalar_method = base_method_name == "scalar"
    if is_scalar_method and model_name and dataset_name:
        if model_name not in TEMPERATURE_SETTINGS:
            TEMPERATURE_SETTINGS[model_name] = {}
        if dataset_name not in TEMPERATURE_SETTINGS[model_name]:
            TEMPERATURE_SETTINGS[model_name][dataset_name] = {}
    
    # For each confidence measure independently, pick the best loss variant
    # This treats loss type as another hyperparameter to optimize over
    for conf_measure in confidence_measures:
        best_loss = None
        criterion = SUB_METRIC_CRITERIA.get(SELECTION_METRIC, "min")
        best_brier = float('inf') if criterion == "min" else float('-inf')
        best_temp_info = None
        
        for loss_name, data in loss_data.items():
            calib_data = data.get("calibration_validation", {})
            if conf_measure in calib_data:
                metric_data = calib_data[conf_measure].get(SELECTION_METRIC, {})
                if isinstance(metric_data, dict) and "mean" in metric_data:
                    metric_mean = metric_data["mean"]
                    is_better = (metric_mean < best_brier if criterion == "min" else metric_mean > best_brier)
                    if is_better:
                        best_brier = metric_mean
                        best_loss = loss_name
                        # Get temperature info for this loss and confidence measure
                        best_temp_info = loss_temp_info[loss_name].get(conf_measure, {})
        
        if best_loss is not None:
            # Copy results from the winning loss variant for this confidence measure
            best_data = loss_data[best_loss]
            
            if conf_measure in best_data.get("calibration_validation", {}):
                final_result["calibration_validation"][conf_measure] = best_data["calibration_validation"][conf_measure].copy()
            
            if conf_measure in best_data.get("test", {}):
                final_result["test"][conf_measure] = best_data["test"][conf_measure].copy()
            
            # Store temperature setting for scalar method
            if is_scalar_method and model_name and dataset_name:
                temp_info = {
                    'loss_type': best_loss,
                    'metric_score': best_brier,
                    'temperature': best_temp_info.get('temperature', None),
                    'run_id': best_temp_info.get('run_id', '')
                }
                
                TEMPERATURE_SETTINGS[model_name][dataset_name][conf_measure] = temp_info
            
            temp_display = f" (T={best_temp_info.get('temperature', '?')})" if best_temp_info.get('temperature') else ""
            print(f"      {conf_measure}: chose {best_loss} loss{temp_display} ({SELECTION_METRIC}: {best_brier:.4f})")
    
    return final_result


def extract_temperature_info_from_loss_folder(loss_folder_path):
    """Extract temperature information from the loss folder by examining the best run data."""
    temp_info = {}
    
    # Look for the final results file which contains info about which run was selected
    results_file = os.path.join(loss_folder_path, "final_overall_results_normalised.json")
    if not os.path.exists(results_file):
        return temp_info
    
    # Look through subfolders to find temperature patterns and match them to the results
    aggregated_files = []
    for root, dirs, files in os.walk(loss_folder_path):
        if "aggregated_normalised_results.json" in files:
            run_id = os.path.relpath(root, loss_folder_path)
            if run_id != ".":  # Skip the root folder
                temp = extract_temperature_from_run_id(run_id)
                if temp is not None:
                    aggregated_files.append((run_id, temp, os.path.join(root, "aggregated_normalised_results.json")))
    
    # Load the final results to see which runs were selected for each confidence measure
    with open(results_file) as f:
        final_data = json.load(f)
    
    # For each confidence measure in the final results, find which temperature was likely used
    # This is a heuristic approach since we don't store this explicitly
    calib_data = final_data.get("calibration_validation", {})
    
    for conf_measure in calib_data:
        final_metric = calib_data[conf_measure].get(SELECTION_METRIC, {})
        if not isinstance(final_metric, dict) or "mean" not in final_metric:
            continue
            
        final_score = final_metric["mean"]
        
        # Find the aggregated file that most likely produced this score
        best_match_temp = None
        best_match_diff = float('inf')
        
        for run_id, temp, agg_file in aggregated_files:
            try:
                with open(agg_file) as f:
                    agg_data = json.load(f)
                
                agg_calib = agg_data.get("calibration_validation", {})
                if conf_measure in agg_calib:
                    agg_scores = agg_calib[conf_measure].get(SELECTION_METRIC, [])
                    if agg_scores:
                        agg_mean = sum(agg_scores) / len(agg_scores)
                        diff = abs(agg_mean - final_score)
                        if diff < best_match_diff:
                            best_match_diff = diff
                            best_match_temp = temp
                            
            except (json.JSONDecodeError, FileNotFoundError):
                continue
        
        if best_match_temp is not None:
            temp_info[conf_measure] = {
                'temperature': best_match_temp,
                'run_id': f"temp_{best_match_temp}"
            }
    
    return temp_info


def process_simple_cross_loss_comparisons(models, dsets):
    """
    Cross-loss comparison treating loss type as another hyperparameter.
    For each confidence measure independently, choose the loss variant (NLL vs SS/adaptive) 
    with the best calibration validation Brier score.
    This gives one final result per method instead of separate results for each loss type.
    """
    methods_with_losses = ["scalar", "head_platt", "head_transformer"]
    single_methods = ["pre_trained_temp_0.5", "pre_trained_temp_1.0"]
    
    for model, dset in itertools.product(models, dsets):
        model_dset_path = f"results/{model}/{dset}"
        overall_results_path = os.path.join(model_dset_path, "overall_results")
        
        print(f"  Cross-loss comparison for {model}/{dset}")
        
        # Process methods with loss variants (treating loss as hyperparameter)
        for base_method in methods_with_losses:
            print(f"    Processing {base_method} (comparing cross_entropy vs adaptive loss)")
            
            # Dynamically discover confidence measures from the data
            loss_variants = {
                "cross_entropy": f"cross_entropy_{base_method}",
                "adaptive": f"adaptive_{base_method}"
            }
            
            all_confidence_measures = set()
            loss_data = {}
            
            # Load data from both loss variants and collect all confidence measures
            for loss_name, loss_folder in loss_variants.items():
                path = os.path.join(model_dset_path, loss_folder, "final_overall_results_normalised.json")
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                        loss_data[loss_name] = data
                        # Collect confidence measures from this variant
                        calib_data = data.get("calibration_validation", {})
                        all_confidence_measures.update(calib_data.keys())
            
            if not loss_data:
                print(f"      No data found for {base_method}")
                continue
            
            # Add standard collapsed names if we have temperature variants
            standard_measures = set()
            for measure in all_confidence_measures:
                if measure.startswith("GIBBS_") and "_MGC_n10" in measure:
                    standard_measures.add("GIBBS_MGC_n10")
                elif measure.startswith("COLD_") and "_MGC_n10" in measure:
                    standard_measures.add("COLD_MGC_n10")
                else:
                    standard_measures.add(measure)
            
            all_confidence_measures = standard_measures
            print(f"      Found confidence measures: {sorted(all_confidence_measures)}")
            
            best_results = simple_cross_loss_comparison(
                model_dset_path, base_method, list(all_confidence_measures), model, dset
            )
            
            if best_results["test"] or best_results["calibration_validation"]:
                output_path = os.path.join(overall_results_path, base_method, "final_overall_results_normalised.json")
                save_overall_results(best_results, output_path)
                print(f"      Saved final results to: {output_path}")
        
        # Copy single methods directly (no loss variants to compare)
        for single_method in single_methods:
            print(f"    Copying {single_method} (no loss variants)")
            
            source_path = os.path.join(model_dset_path, single_method, "final_overall_results_normalised.json")
            if os.path.exists(source_path):
                with open(source_path) as f:
                    data = json.load(f)
                
                output_path = os.path.join(overall_results_path, single_method, "final_overall_results_normalised.json")
                save_overall_results(data, output_path)
                print(f"      Copied to: {output_path}")


# ────────────────────────────  main  ─────────────────────────────── #

if __name__ == "__main__":

    models = [
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Ministral-8B-Instruct-2410"
    ]
    files = [
        "pre_trained_temp_0.5",
        "pre_trained_temp_1.0",
        "cross_entropy_scalar",
        "adaptive_scalar",
        "cross_entropy_head_platt",
        "adaptive_head_platt",
        "cross_entropy_head_transformer",
        "adaptive_head_transformer",
    ]
    
    dsets = [
        "natural_questions",
        "trivia_qa",
        "squad"
    ]

    # First process individual methods with independent hyperparameter selection per confidence measure
    print(f"Step 1: Processing individual methods with independent hyperparameter selection per confidence measure...")
    print(f"  Selection metric: {SELECTION_METRIC}")
    print("  For each confidence measure independently:")
    print("  - Look at hyperparameter configurations in subfolders")  
    print(f"  - Find best {SELECTION_METRIC} score on calibration validation set for that measure")
    print("  - Store corresponding test results for that measure and method")
    print("  - For Gibbs/Cold measures: collapse multiple temperature settings into best one")
    print()
    
    for model, dset, file in itertools.product(models, dsets, files):
         base_folder   = f"results/{model}/{dset}/{file}"
         print(f"    Processing: {base_folder}")
         out_json = process_files_recursive(base_folder, metric_type="normalised", 
                                           model_name=model, dataset_name=dset)

         save_overall_results(
             out_json,
             os.path.join(base_folder, "final_overall_results_normalised.json"),
         )
    
    # Then process cross-loss comparisons treating loss as another hyperparameter
    print("\nStep 2: Collapsing best results by treating loss type as another hyperparameter...")
    print("  For each confidence measure independently:")
    print("  - Compare NLL vs SS (adaptive) loss variants")
    print("  - Choose loss type with best calibration validation Brier score")
    print("  - This gives one result per method (one temperature scaling, one Platt, etc.)")
    print()
    
    process_simple_cross_loss_comparisons(models, dsets)
    
    # Generate temperature settings table for TS method
    print("\nStep 3: Generating temperature settings table for Temperature Scaling...")
    save_temperature_settings_table()