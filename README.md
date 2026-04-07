
# BodoLegalQA- Bodo Question Answering Experiment
This repository contains the code for the Bodo Question Answering Experiment. The code is organized into different folders based on the model used for training and evaluation.

## Project Structure
- `model/`: This folder contains the code for training and evaluating different models on the Bodo Question Answering dataset. Each model has its own subfolder named after the model used (e.g., `google/mt5-small`, `google-bert/bert-base-multilingual-cased`, etc.).
- `data/`: This folder contains the training and testing datasets in JSON format.
- `output/`: This folder will be used to save the trained models and evaluation results.

## Requirements
- Python 3.10+
- transformers 4.46.3
- NVIDIA GPU with CUDA support (tested on A6000)
- ~10GB VRAM minimum (48GB recommended for large batch sizes)

## Installations

1. Clone the repository:
```bash
git clone https://github.com/mwnthainarzary/bodo-legal-qa.git
cd bodo-legal-qa
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install transformers==4.46.3 \ 
       accelerate==0.26.0 \
       torch sentencepiece protobuf \
       evaluate sacrebleu rouge_score bert_score \
       scikit-learn matplotlib numpy
```

## Usage

### Training

Run Script:

python model/dataset_name{ai4bharat/iiith}/model.py \
       --train_path  data/train.json \
       --test_path   data/test.json  \
       --output_dir  output   \
       --model_name  model_name \
       --epochs      20

List of Model Used:
1. distilbert/distilbert-base-multilingual-cased
2. google-bert/bert-base-multilingual-cased
3. google-bert/bert-base-multilingual-uncased
4. ai4bharat/indic-bert
5. ai4bharat/IndicBERTv2-MLM-Back-TLM
6. ai4bharat/IndicBART
7. FacebookAI/xlm-roberta-base
8. FacebookAI/xlm-roberta-large
9. google/muril-base-cased
10. google/muril-large-cased

### Output Folder
# Output files in output_dir/:
- hyperparameters.txt    model config, data paths, all training settings
- epoch_eval_log.txt     eval_loss + span_acc per epoch (live updates)
- train_loss_graph.png   Train Loss vs Epochs graph (PNG)
- evaluation_scores.txt  BLEU, ROUGE, BERTScore, Cosine Sim, Exact Match
- test_results.txt       per-sample: question, expected, predicted, cosine
- final_model/           saved IndicBERT weights + SentencePiece tokenizer

## Evaluation Metrics
- BLEU Score: Evaluates n-gram overlap between predicted and reference answers.
- ROUGE Score: Measures recall of n-grams, word sequences, and word pairs.
- BERTScore: Uses contextual embeddings to evaluate semantic similarity.
- Cosine Similarity: Measures cosine similarity between predicted and reference answer embeddings.
- Exact Match: Checks if the predicted answer exactly matches the reference answer.

### GPU Optimizations

The script is optimized for NVIDIA Ampere GPUs (A6000, A100, RTX 30/40 series):
- TF32 enabled for faster matrix operations
- BF16 mixed precision training
- Fused AdamW optimizer
- Parallel data loading with prefetching


## Models

| Model | Base | Parameters | Description |
|-------|------|------------|-------------|
| google/mt5-small | mT5 | 300M | Multilingual T5 model, good for sequence-to-sequence tasks. |
| google-bert/bert-base-multilingual-cased | BERT | 110M
| Multilingual BERT with cased text, suitable for languages with case sensitivity. |
| google-bert/bert-base-multilingual-uncased | BERT | 110M | Multilingual BERT with uncased text, suitable for languages without case sensitivity. |
| ai4bharat/indic-bert | BERT | 110M | IndicBERT is a multilingual BERT model trained on Indian languages, including Bodo. |
| ai4bharat/IndicBERTv2-MLM-Back-TLM | BERT | 110M | An improved version of IndicBERT with additional training on masked language modeling and translation language modeling tasks. |
| ai4bharat/IndicBART | BART | 400M | A sequence-to-sequence model based on BART architecture, trained on Indian languages. |
| FacebookAI/xlm-roberta-base | XLM-RoBERTa | 270M | A multilingual RoBERTa model trained on 100 languages, including Bodo. |

## Results
seet the results in RESULT.md file for detailed evaluation scores and comparisons between models on the Ai4Bharat and IIITH datasets.


## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

