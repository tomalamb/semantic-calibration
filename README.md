# Improving Semantic Uncertainty Quantification in Language Model Question-Answering via Token-Level Temperature Scaling

## Abstract

Calibration is central to reliable semantic uncertainty quantification, yet prior work has largely focused on discrimination, neglecting calibration. As calibration and discrimination capture distinct aspects of uncertainty, focusing on discrimination alone yields an incomplete picture. We address this gap by systematically evaluating both aspects across a broad set of confidence measures. We show that current approaches, particularly fixed-temperature heuristics, produce systematically miscalibrated and poorly discriminative semantic confidence distributions. We demonstrate that optimising a single scalar temperature, which, we argue, provides a suitable inductive bias, is a surprisingly simple yet effective solution. Our exhaustive evaluation confirms that temperature scaling consistently improves semantic calibration, discrimination, and downstream entropy, outperforming both heuristic baselines and more expressive token-level recalibration methods on question-answering tasks.

### Motivation 

![Figure 1 placeholder](figures/pullfigure.png)

**Temperature Scaling Improves Semantic Uncertainty Quantification.** We compare the same base model with different temperature parameters, each generating ten responses for the same input, and cluster responses into semantic groups. We compute the semantic confidence measure introdcued by Kuhn et al. (2023). Panel (a) uses the recommended temperature of 0.5. Panel (b) uses a temperature optimized on a calibration set. Optimised temperature scaling offers a simple way to improve both semantic calibration and discrimination.

### Semantic Confidence Measures 

![Figure 2 placeholder](figures/methodology.png)

We evaluate the calibration and discrimination of semantic confidence measures across multiple question-answering datasets, including TriviaQA, Natural Questions, and SQuAD, using popular instruction-tuned language models. To investigate the effect of optimising temperature parameters (**Temperature Scaling (TS)**) on semantic uncertainty quantification, we compare against several baseline post-hoc, token-level recalibration techniques: an **Adaptive Temperature Scaling (ATS)** head that predicts token-specific temperatures, **Platt Scaling** with a diagonal affine logit transform, and fixed-temperature baselines of τ = 1.0 (Base) and τ = 0.5 (SE). We compare how each method influences semantic calibration, discrimination, and uncertainty across existing and novel semantic confidence measures.

### Main Results

![Figure 2 placeholder](figures/calibration_scatter_best.png)

Optimised **Temperature Scaling (TS)** consistently improves semantic uncertainty quantification, outperforming both fixed-temperature heuristics (Base and SE) used in prior work, and more complex calibration methods such as **Adaptive Temperature Scaling (ATS)** and **Platt Scaling**. Improvements hold across all question-answering datasets, demonstrating that TS provides a simple, robust, and effective means of enhancing both **semantic calibration** and **discrimination** of semantic confidence measures.

## Quick start

Requirements: Python 3.10+, PyTorch and dependencies listed in `requirements.txt`. Create and activate a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Run a short help check for the head training script:

```bash
python3 src/head_temp_training.py --help
```

## Reproducing key results

1. Prepare dataset(s) and update paths in `configs/`.
2. Train calibration head (token-level) with `src/head_temp_training.py`.
3. Evaluate calibration using scripts in `plotting/` and `src/utils/`.

