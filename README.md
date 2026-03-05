
<h1 align="center">Seeing Clearly without Training</h1>

<p align="center"><strong>Mitigating Hallucinations in Multimodal LLMs for Remote Sensing</strong></p>

<p align="center">
  <b>Yi Liu</b><sup>1,2</sup> &nbsp; <b>Jing Zhang</b><sup>1,2</sup><code>†</code> &nbsp; <b>Di Wang</b><sup>1,2</sup><code>†</code> &nbsp; <b>Xiaoyu Tian</b><sup>3</sup> &nbsp; <b>Haonan Guo</b><sup>1,2</sup> &nbsp; <b>Bo Du</b><sup>1,2</sup><code>†</code>
</p>


<p align="center">
  <sup>1</sup> School of Computer Science, Wuhan University, China &nbsp;&nbsp;|&nbsp;&nbsp; <sup>2</sup> Zhongguancun Academy, China &nbsp;&nbsp;|&nbsp;&nbsp; <sup>3</sup> School of Computer Science, Chongqing University, China
</p>
<p align="center">
  <small><code>†</code> Corresponding Author</small>
</p>


<p align="center">
  <a href="https://github.com/MiliLab/RADAR"><img src="https://img.shields.io/badge/🌐-Project%20Page-6c757d?style=flat-square" alt="Project"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2603.02754"><img src="https://img.shields.io/badge/arXiv-2603.02754-b31b1b?style=flat-square" alt="arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/LIUYIfasdf/RSHBench"><img src="https://img.shields.io/badge/📊-Data%20(RSHBench)-ffcc00?style=flat-square" alt="Data"></a>
</p>


<p align="center">
  If you find this project helpful, please consider giving it a <strong>⭐ star</strong>!
</p>

---

## 🔥 News

- **[2025]** We release the code and **RSHBench** benchmark for hallucination diagnosis in RS-VQA.
- **[2025]** We propose **RADAR** (Relative Attention-Driven Actively Reasoning), a training-free inference method that reduces factual and logical hallucinations via query-conditioned relative attention and progressive evidence acquisition.

---

## 📖 Introduction

Multimodal large language models (MLLMs) suffer from pronounced **hallucinations** in remote sensing visual question-answering (RS-VQA), mainly due to: (1) **Type 1 — Cannot find:** attention becomes diffuse and misses the target region; (2) **Type 2 — Cannot see clearly:** the model attends the right area but fails at fine-grained recognition. To address this, we introduce:

- **RSHBench** — A protocol-driven benchmark for fine-grained diagnosis of **factual** and **logical** hallucinations in RS-VQA, with standardized generation and multi-judge evaluation.
- **RADAR** — A **training-free** inference framework that uses **Query-Conditioned Relative Attention (QCRA)** to guide a two-stage zoom-in: *where*-oriented localization followed by *what*-oriented fine-grained evidence refinement, with a focus test to avoid cropping when attention is diffuse.

Extensive experiments show that RADAR consistently improves RS-VQA accuracy (e.g. +2%–4% on representative benchmarks) and reduces hallucination rates (e.g. ~10% reduction on RSHBench).

### Query-Conditioned Relative Attention (QCRA)

Given an image and task-focused query vs. global-comprehension query, we derive layer-wise relative attention and aggregate top-$k$ layers to produce a query-conditioned heatmap for region selection.
<img width="3600" height="1617" alt="method" src="https://github.com/user-attachments/assets/0f46e4a6-caa8-41fd-aefa-0df4ec0d95d9" />


*Figure 1: QCRA pipeline — relative attention contrast and multi-scale evidence construction.*

---

## 📂 Repository Structure

```
msswift/
├── RSHBench/
│   ├── infer.py         # Run model inference (reasoning + answer)
│   ├── eval.py          # Multi-judge hallucination evaluation
│   ├── score.py         # Aggregate HR and subtype statistics
│   └── score_judge.py   # Judge reliability (LOO, agreement)
├── prompt/              # COT, hallucination judge prompts
├── infer_qwen.py        # Main inference with RADAR 
├── infer_llava.py       # RADAR inference for LLaVA 
├── qwen_methods.py      # QCRA & RADAR logic for Qwen-VL
├── llava_methods.py     # QCRA & RADAR logic for LLaVA 
├── model_infer.sh       # Multi-GPU parallel inference
├── add_chunk.py         # Merge chunked inference results
├── get_score.py         # VQA accuracy scoring 
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/MiliLab/RADAR.git
cd RADAR

# Create environment (Python 3.10 recommended)
conda create -n rshbench python=3.10
conda activate rshbench

# Install dependencies (transformers, torch, PIL, etc.)
pip install torch torchvision transformers pillow numpy tqdm
# For Qwen2-VL: pip install qwen-vl-utils
# For LLaVA/OneVision: ensure swift and compatible transformers are installed
```

---

## 💡 Demo & Usage

### Run RADAR Inference 

** Qwen-VL:**

```bash
python infer_qwen.py \
    --model qwen3_4b \
    --task MME-RealWorld-RS \
    --att 640 \
    --total_chunks 1 \
    --chunk_id 0 \
    --save_path ./outputs \
    --stage stage1 \
```

**LLaVA / LLaVA-OneVision:**

```bash
python infer_llava.py \
    --model_name your_llava_model \
    --task MME-RealWorld-RS \
    --save_path ./outputs
```

### Key Options

| Option                          | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `--model`                       | Model alias (e.g. `qwen3_4b`, `geozero`)         |
| `--task`                        | Dataset: `lhrs`, `lrsbench`, `MME_RealWorld`etc. |
| `--Image_size`                  | Max side length for attention/resize (e.g. 640)  |
| `--total_chunks` / `--chunk_id` | Data sharding for multi-GPU                      |

RADAR is **training-free**: it uses the model’s internal attention to compute QCRA, runs a focus test, and optionally crops to question-relevant regions before generating the final answer.

### Multi-GPU Inference

Edit `model_infer.sh` (GPU IDs, chunks, model, task), then:

```bash
bash model_infer.sh
```

Merge chunk results with `add_chunk.py` if needed.

### RSHBench: Hallucination Evaluation Pipeline

1. **Inference** — Generate reasoning + answer for each sample:

   ```bash
   python RSHBench/infer.py \
       --dataset path/to/dataset.json \
       --model_name your_model_name \
       --output_dir outputs/
   ```

2. **Multi-judge evaluation** — Annotate hallucination (binary + taxonomy):

   ```bash
   python RSHBench/eval.py \
       --input outputs/infer.jsonl \
       --judge_model judge_name \
       --output outputs/eval.jsonl
   ```

3. **Aggregate results** — HR and subtype statistics:

   ```bash
   python RSHBench/score.py --input outputs/eval.jsonl
   ```

4. **Judge reliability (optional):**

   ```bash
   python RSHBench/score_judge.py --input outputs/eval.jsonl
   ```

---

## 🧠 Hallucination Taxonomy (RSHBench)

- **Factual:** OBJ (object/category), ATT (attribute), SPA (spatial/relational).
- **Logical:** IR (invalid reasoning), CI (unjustified causality), INC (internal inconsistency), SO (semantic over-attribution).

Hallucination rate (HR) and subtype statistics are computed over the evaluation set; consensus can be obtained via majority vote across judges.

---

## 🌟 Evaluation

We evaluate RADAR on:

- **LRS-VQA** (FAIR, Bridge, STAR) — large-scale RS imagery reasoning  
- **MME-RealWorld-RS** (Position, Color, Count) — localization and attribute discrimination  
- **LHRS-Bench** — recognition, spatial perception, and reasoning  
- **RSHBench** — hallucination rate (HR) and fine-grained factual/logical breakdown  

RADAR consistently improves accuracy on these benchmarks and reduces both factual and logical hallucinations. Below are key tables from the paper.

---

### Judge agreement (LOO) on RSHBench

Leave-one-out agreement of expert judges for the binary hallucination decision (Accuracy, Cohen's κ, MCC).

| Judge        | Accuracy   | Cohen's κ  | MCC        |
| ------------ | ---------- | ---------- | ---------- |
| Gemini-3-pro | 0.7882     | 0.5726     | 0.5770     |
| GPT-5.2      | **0.9288** | **0.8553** | **0.8591** |
| Qwen3-max    | 0.9045     | 0.8058     | 0.8070     |

---

### Hallucination evaluation on RSHBench (consensus)

All values are percentages. **HR** = overall hallucination rate; **HR_F** = Factual, **HR_L** = Logical. Subtypes: OBJ, ATT, SPA (factual); IR, CI, INC, SO (logical).

| Models            | OBJ       | ATT       | SPA       | HR_F      | IR        | CI       | INC      | SO       | HR_L      | **HR**    |
| ----------------- | --------- | --------- | --------- | --------- | --------- | -------- | -------- | -------- | --------- | --------- |
| *Closed-source*   |           |           |           |           |           |          |          |          |           |           |
| Claude-3-7        | 40.70     | 28.30     | 11.32     | 55.53     | 20.49     | 0.27     | 1.08     | 14.29    | 24.53     | 56.33     |
| Gemini-2.5-pro    | 33.42     | 31.54     | 15.36     | 48.79     | 27.49     | 0.54     | 0.81     | 15.09    | 29.65     | 49.06     |
| GPT-4o            | 30.19     | 26.68     | 11.32     | 46.90     | **19.41** | 0.27     | **0.27** | 11.05    | **21.02** | 47.44     |
| *Open-source*     |           |           |           |           |           |          |          |          |           |           |
| GLM-4.6v          | 35.31     | **21.02** | **9.16**  | 48.79     | 21.02     | **0.00** | 0.54     | **9.16** | 22.91     | 49.60     |
| LLaVA-1.5-7B      | **25.88** | 29.11     | 14.29     | 46.90     | 25.61     | 2.96     | 2.70     | 16.71    | 26.95     | 47.71     |
| Qwen3-VL-4B       | 44.47     | 31.81     | 14.82     | 61.19     | 29.92     | 0.27     | 2.43     | 15.63    | 34.77     | 61.19     |
| LLaMA-3.2-90B     | 35.85     | 25.88     | 12.13     | 51.48     | 26.15     | 0.27     | 3.77     | 15.36    | 29.11     | 52.02     |
| GeoZero           | 33.42     | 33.96     | 15.90     | 49.87     | 28.30     | 3.77     | 2.16     | 18.06    | 29.65     | 49.87     |
| **GeoZero+RADAR** | **28.03** | **25.61** | **13.48** | **38.54** | 21.83     | 2.16     | 1.89     | 15.63    | 24.80     | **38.81** |

---

### Overall accuracy (%) on RS-VQA benchmarks

Accuracy on LRS-VQA (FAIR, Bridge, STAR, AA), MME-RealWorld-RS (Position, Color, Count, AA), LHRS-Bench. **Avg.** = mean across the three benchmarks.

| Methods             | FAIR      | Bridge    | STAR      | LRS-VQA AA | Position  | Color     | Count     | MME AA    | LHRS-Bench | **Avg.**  |
| ------------------- | --------- | --------- | --------- | ---------- | --------- | --------- | --------- | --------- | ---------- | --------- |
| Llama-4-scout       | 19.72     | 29.19     | 23.73     | 24.21      | 26.70     | 23.72     | 15.64     | 22.02     | 37.33      | 27.85     |
| GPT-4o              | 22.89     | 24.39     | 29.78     | 25.69      | 36.37     | 32.43     | 15.89     | 28.23     | 66.19      | 40.04     |
| Qwen3-VL-4B         | 26.23     | 29.47     | 32.86     | 29.52      | 55.53     | 40.24     | 7.59      | 34.45     | 65.35      | 43.11     |
| Qwen3-VL-8B         | 29.71     | 32.77     | **35.01** | 32.49      | 54.97     | 46.06     | 14.44     | 38.49     | 66.03      | 45.67     |
| GeoChat             | 20.18     | 24.54     | 13.75     | 19.49      | 25.06     | 23.11     | 15.66     | 21.28     | 37.62      | 26.13     |
| GeoZero             | 29.53     | 31.26     | 33.96     | 31.58      | 57.04     | 44.30     | 15.74     | 39.03     | 66.08      | 45.56     |
| **RADAR (GeoZero)** | **31.21** | **33.33** | 34.11     | **32.88**  | **58.15** | **50.52** | **20.47** | **43.05** | **67.47**  | **47.40** |

---

### RADAR vs ViCrop (accuracy %, gains in parentheses)

| Method             | LRS-VQA FAIR      | Bridge            | STAR              | MME Position      | Color              | Count             | LHRS-Bench        |
| ------------------ | ----------------- | ----------------- | ----------------- | ----------------- | ------------------ | ----------------- | ----------------- |
| Qwen3-VL           | 26.23             | 29.47             | 32.86             | 55.53             | 40.24              | 7.59              | 65.35             |
| + ViCrop           | 27.46             | 28.91             | 33.31             | 54.57             | 42.87              | 10.60             | 63.51             |
| **+ RADAR (Ours)** | **30.77 (+4.54)** | 29.94 (+0.47)     | **34.98 (+2.13)** | 56.64 (+1.11)     | **53.71 (+13.47)** | **15.01 (+7.42)** | **67.73 (+2.38)** |
| GeoZero            | 29.53             | 31.26             | 33.96             | 57.04             | 44.30              | 15.74             | 66.08             |
| + ViCrop           | 28.83             | 31.73             | 32.91             | 57.04             | 47.41              | 18.19             | 65.14             |
| **+ RADAR (Ours)** | **31.21 (+1.68)** | **33.33 (+2.07)** | 34.11 (+0.15)     | **58.15 (+1.11)** | **50.52 (+6.22)**  | **20.47 (+4.73)** | **67.47 (+1.39)** |

---

### Ablation: two-stage RADAR (Qwen3-VL-4B)

| Configuration     | MME-RealWorld-RS | LHRS-Bench |
| ----------------- | ---------------- | ---------- |
| Baseline          | 34.45            | 65.36      |
| RADAR w/o Stage 2 | 39.05            | 66.17      |
| RADAR w/o Stage 1 | 38.88            | 66.87      |
| **RADAR (full)**  | **41.79**        | **67.73**  |

---

### Qualitative examples

QCRA heatmaps from *where*-oriented (Stage1) and *what*-oriented (Stage2) queries; dashed boxes mark regions selected for zoom-in evidence extraction.


<img width="3271" height="1327" alt="show" src="https://github.com/user-attachments/assets/25a07ffe-8b16-4d14-9306-3e88977a424f" />

*Figure 3: Qualitative examples of RADAR's progressive evidence refinement.*

## 📊 Data

RSHBench evaluation set and related data:

- **Dataset:** [RSHBench on Hugging Face](https://huggingface.co/datasets/LIUYIfasdf/RSHBench)

---

## 📜 Citation

If you use this code or RSHBench in your research, please cite:

```bibtex
@article{liu2026radar,
  title={Seeing Clearly without Training: Mitigating Hallucinations in Multimodal LLMs for Remote Sensing},
  author={Liu, Yi and Zhang, Jing and Wang, Di and Tian, Xiaoyu and Guo, Haonan and Du, Bo},
  journal={arXiv preprint arXiv:2603.02754},
  year={2026},
  doi={10.48550/arXiv.2603.02754}
}
```

---

## 🤝 Acknowledgements

This work is supported by Wuhan University, Zhongguancun Academy, and Chongqing University. We thank the communities behind LRS-VQA, MME-RealWorld-RS, LHRS-Bench, and related MLLM and remote sensing benchmarks.

