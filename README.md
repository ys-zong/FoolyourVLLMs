# Fool your (V)LLMs
Code for [paper](https://arxiv.org/abs/2310.01651): Fool Your (Vision and) Language Model With Embarrassingly Simple Permutations

## Content
- [Environment](#environment)
  - [Data](#data)
- [Usage](#usage)
  - [LLM MCQ](#llm-mcq)
  - [VLLM MCQ](#vllm-mcq)
- [Citation](#citation)

## Environment
```bash
conda create -n MCQ python=3.9
conda activate MCQ
pip install -r requirements.txt
```

Additionally, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA) and [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl) repo for specific installation.

### Data
For language task, download MMLU dataset from [here](https://github.com/hendrycks/test) and put it in `data/MMLU/` directory.

For vision-language task, download ScienceQA dataset from [here](https://github.com/lupantech/ScienceQA#ghost-download-the-dataset) and put it in `data/ScienceQA/` directory. Follow `data/preprocess_ScienceQA.ipynb` for preprocessing. Only questions with image input are used.

## Usage
### LLM MCQ
1. Original option order:
```bash
python LLMs_attack.py --data_dir ./data/MMLU/ --engine vicuna7b
```

2. Permute option orders:
```bash
python LLMs_attack.py --data_dir ./data/MMLU/ --engine vicuna7b --permutation_attack
```

3. Reduce the number of options:
```bash
python LLMs_attack.py --data_dir ./data/MMLU/ --engine vicuna7b --n_reduced 2
```

4. Reduce the number of options and then permute:
```bash
python LLMs_attack.py --data_dir ./data/MMLU/ --engine vicuna7b --n_reduced 2 --reduce_attack
```

5. Move all ground-truth answers to certain position:
```bash
python LLMs_attack.py --data_dir ./data/MMLU/ --engine vicuna7b --position_permute
```

### VLLM MCQ
1. Original option order:
```bash
python VLLMs_attack.py --data_dir ./data/ScienceQA/ --engine instructblip7b
```

2. Permute option orders:
```bash
python VLLMs_attack.py --data_dir ./data/ScienceQA/ --engine instructblip7b --permutation_attack
```

3. Reduce the number of options:
```bash
python VLLMs_attack.py --data_dir ./data/ScienceQA/ --engine instructblip7b --n_reduced 2
```

4. Reduce the number of options and then permute:
```bash
python VLLMs_attack.py --data_dir ./data/ScienceQA/ --engine instructblip7b --n_reduced 2 --reduce_attack
```

5. Move all ground-truth answers to certain position:
```bash
python VLLMs_attack.py --data_dir ./data/ScienceQA/ --engine instructblip7b --position_permute
```


### Arguments:
- `--ntrain`: number of in-context demonstrations.
- `--data_dir`: path to the dataset.
- `--engine`: which model to use (can use multiple).
- `--n_reduced`: specifies the reduced total number of options.
- `--reduce_attack`: permute the options after reduction.
- `--use_subset`: use subset to test.
- `--permutation_attack`: adversarial permutation to the options.
- `--position_permute`: move all GT answers to certain position (A/B/C/D).
- `--load_in_8bit`: (optional) 8 bit loading to fit large models into GPU memory.

## Citation
```
@article{zong2023fool,
  title={Fool Your (Vision and) Language Model With Embarrassingly Simple Permutations},
  author={Zong, Yongshuo and Yu, Tingyang and Zhao, Bingchen and Chavhan, Ruchika and Hospedales, Timothy},
  journal={arXiv preprint arXiv:2310.01651},
  year={2023}
}
```