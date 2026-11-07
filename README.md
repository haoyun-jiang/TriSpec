# [NeurIPS 2026] TriSpec

This is the official repository of "TriSpec: Ternary Speculative Decoding via Lightweight Proxy Verification". It provides implementations of TriSpec built on HASS and EAGLE-3, including draft model training, proxy verification, and benchmark evaluation.

[Paper (arXiv)](https://arxiv.org/abs/2601.23180)

## 🛠 Installation

To set up the environment, please ensure you have Python installed, then run:

```bash
# Clone the repository
git clone https://github.com/haoyun-jiang/TriSpec.git
cd TriSpec

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Draft Model Training

Update the model, data, and checkpoint paths in the shell scripts before running. Each command block is a separate experiment; run the blocks you need, or comment out the others.

The training process for the draft model consists of three main phases:

### Phase 1: Target Model Autoregressive Data Generation
In this step, we use the target model to generate autoregressive data for subsequent training.

```bash
# Start the server
bash ge_autodata/server.sh
# Start the generation process
bash ge_autodata/generate.sh
```

### Phase 2: Offline Hidden States Generation
Generate the hidden states required for the draft model's training.

```bash
bash ge_data_assistant.sh
```

### Phase 3: Training
Depending on the architecture you want to use, run one of the following commands:

*   **For Hass-based draft model:**
    ```bash
    bash train_hass_baseline_trispec.sh
    ```
*   **For Eagle3-based draft model:**
    ```bash
    bash train_eagle3_baseline_trispec.sh
    ```


## 📊 Evaluation

Once the training is complete, you can evaluate the performance using the following scripts:

### Evaluate Baseline
To run the standard baseline evaluation:
```bash
bash eval_baseline.sh
```

### Evaluate TriSpec
To run the evaluation for the proposed TriSpec method:
```bash
bash eval_trispec.sh
```

## 📜 Citation

If you find this work useful for your research, please cite our [paper](https://arxiv.org/abs/2601.23180):

```bibtex
@article{jiang2026trispec,
  title={TriSpec: Ternary Speculative Decoding via Lightweight Proxy Verification},
  author={Jiang, Haoyun and He, Junqi and Hong, Feng and Yang, Xinlong and Zhang, Jianwei and Li, Zheng and Zhuge, Zhengyang and Chen, Zhiyong and Han, Bo and Lin, Junyang and Yao, Jiangchao},
  journal={arXiv preprint arXiv:2601.23180},
  year={2026},
  doi={10.48550/arXiv.2601.23180},
  url={https://arxiv.org/abs/2601.23180}
}
```
