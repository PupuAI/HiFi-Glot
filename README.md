<p align="center">
<img src='images/logo.png' style="text-align: center; width: 150px" >
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](TBD)
[![Project Page](https://img.shields.io/badge/HiFiGlot-Website-green)](https://www.yichenggu.com/HiFi-Glot/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Aalto-Speech-Synthesis/HiFi-Glot)

</div>

# HiFi-Glot: High-Fidelity Neural Formant Synthesis with Differentiable Resonant Filters

This is the official Hugging Face model repository for the paper "[HiFi-Glot: High-Fidelity Neural Formant Synthesis with Differentiable Resonant Filters](TBD)". which is the first end-to-end neural formant synthesis system that achieves high perceptual quality and precise formant control, as shown below:

<br>
<div align="center">
<img src="images/model.png" width="90%">
</div>
<br>

This script provides an example of how to set up the environment, inference from our pretrained [checkpoints](https://huggingface.co/Aalto-Speech-Synthesis/HiFi-Glot), and train/finetune your own models with custom datasets.

> **NOTE:** You need to run every command of this recipe in the `HiFi-Glot` root path:
> ```bash
> cd HiFi-Glot
> ```

## 📀 Installation

```bash
git clone https://github.com/PupuAI/HiFi-Glot.git
cd HiFi-Glot

# Install Python Environment
conda create --name hifiglot python=3.12
conda activate hifiglot

# Install Python Packages Dependencies
pip install -r requirements.txt
pip install -e .
```

## 🐍 Usage in Python

### 1. Data Preparation

By default, we utilize the AIShell3 dataset for training. You can download the dataset and specify their root directory in `dataset/train.json`:

```json
// TODO: Fill in the root directory
{
    "name": "AIShell",
    "path": "{PATH_TO_DATASET}/",
}
```

Our codebase supports file-based custom datasets. To add your own dataset, list out all the audio files in the dataset root directory and save them to `dataset/filelist/custom_dataset.json`:

```json
// TODO: List all the audio files in your dataset
[
    {
        "path": "{PARENT_FOLDER_TO_FILE}/{UID}.wav"
    },
    ... ...
    {
        "path": "{PARENT_FOLDER_TO_FILE}/{UID}.wav"
    }
]
```

After that, create a metadata file in `dataset/train_custom.json` to specify the datasets you want to utilize:

```json
// TODO: List all the datasets you want to utilize
{
"datasets":[
    {
        "name": "{CUSTOM_DATASET}",
        "path": "{PATH_TO_DATASET}/{CUSTOM_DATASET}"
    },
    ... ...
    {
        "name": "{CUSTOM_DATASET}",
        "path": "{PATH_TO_DATASET}/{CUSTOM_DATASET}"
    }
],

"filelist_path": "dataset/filelist"

}
```

### 2. Training

Adjust the `configs/{model}/config_{module}.json` to specify the hyperparameters:

```json
{
  ... ...
  // TODO: Choose a suitable batch size, training epoch, and save stride
  "batch_size": 32,
  ... ...
}
```

Then, run the training script like this:

```bash
accelerate launch \
    --num_processes={num_gpus} \
    --num_machines={num_machines} \
    train_e2e_{model}.py \
    --checkpoint_path="experiments/{model}" \
    --config="configs/{model}/config_hifigan.json" \
    --fm_config="configs/{model}/config_feature_map.json"
```

If you want to resume or finetune from a pretrained model, run:

```bash
accelerate launch \
    --num_processes={num_gpus} \
    --num_machines={num_machines} \
    train_e2e_{model}.py \
    --checkpoint_path="experiments/{model}" \
    --config="configs/{model}/config_hifigan.json" \
    --fm_config="configs/{model}/config_feature_map.json" \
    --resume_path="{resume_path}"
```

> **NOTE:** For multi-gpu training, the `main_process_port` is set as `29500` in default. You can change it by specifying such as `--main_process_port 29501`.

### 3. Inference

Pretrained checkpoints, which use the same training set as the paper with more training steps, are released [here](https://huggingface.co/Aalto-Speech-Synthesis/HiFi-Glot).

By default, we provide evaluation samples in `samples/`.  Here is an example for the default scenario.

```bash
python inference_{model}.py \
    --input_path="example" \
    --output_path="example" \
    --config="configs/{model}/config_hifigan.json" \
    --fm_config="configs/{model}/config_feature_map.json" \
    --checkpoint_path="experiments/{model}/{checkpoint}/model.safetensors" \
    --scale_list=[{F1_shift_scale},{F2_shift_scale},{F3_shift_scale}]
```

## ©️ License

Our project is under the [MIT License](LICENSE). It is free for both research and commercial use cases.

## 📚 Citation

```bibtex
@article{afgen,
  title        = {HiFi-Glot: High-Fidelity Neural Formant Synthesis with Differentiable Resonant Filters},
  author       = {TBD},
  year         = {2025},
  journal      = {TBD},
}
```
