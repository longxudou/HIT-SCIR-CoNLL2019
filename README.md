# HIT-SCIR CoNLL2019 Unified Transition-Parser

This repository accompanies the paper, "HIT-SCIR at MRP 2019: A Unified Pipeline for Meaning Representation Parsing via Efficient Training and Effective Encoding", providing codes to train models and pre/post-precessing mrp dataset.

CoNLL2019 Shared Task Official Website: <http://mrp.nlpl.eu/>

## Pre-requisites

- Python 3.6
- JAMR
- NLTK
- Gensim
- Penman
- AllenNLP 0.9.0

For JAMR installation, please refer to https://github.com/DreamerDeo/HIT-SCIR-CoNLL2019/issues/2. 

## Dataset

Total training data is available at [mrp-data].

## Model
Download model from [google-drive] (CoNLL2019 Submission Version). 

For prediction, please specify the BERT path in `config.json` to import the bert-indexer and bert-embedder. More prediction commands could be found in `bash/predict.sh`.

About BERT version, DM/PSD/UCCA/EDS use cased_L-12_H-768_A-12 (`cased-bert-base`) and AMR uses wwm_cased_L-24_H-1024_A-16 (`wwm-cased-bert-large`).

## Usage

### Prepare data

#### Step 1: Add companion to raw data.

We use conllu format companion data. This command adds `companion.conllu` to `data.mrp` and outputs to `data.aug.mrp`

```shell script
python3 toolkit/augment_data.py \
    companion.conllu \
    data.mrp \
    data.aug.mrp
```

For evaluation data, you need to convert udpipe to conllu format and split raw input to 5 files. Run this command instead. 

```shell script
python3 toolkit/preprocess_eval.py \
    udpipe.mrp \
    input.mrp \
    --outdir /path/to/output
```

#### Step 2 (only for AMR): Convert data to amr format and run TAMR aligner.

Different from the other 4 parsers, our AMR parser accepts input of augmented amr format instead of mrp format.

Since TAMR's alignment is built on the JAMR alignment results, you need to set JAMR and CDEC path in `bash/amr_preprocess.sh` and run the command below.

```shell script
bash bash/amr_preprocess.sh \
    data.aug.mrp \
    /path/to/word2wec
```

The final output is `data.aug.mrp.actions.aug.txt` which can be input to AMR parser. 

According to TAMR, it is recommended to use the glove.840B.300d and filter the embeddings by the words and concepts (trimming the tail in word sense) in the data.

### Train the parser

Based on AllenNLP, the training command is like

```shell script
CUDA_VISIBLE_DEVICES=${gpu_id} \
TRAIN_PATH=${train_set} \
DEV_PATH=${dev_set} \
BERT_PATH=${bert_path} \
WORD_DIM=${bert_output_dim} \
LOWER_CASE=${whether_bert_is_uncased} \
BATCH_SIZE=${batch_size} \
    allennlp train \
        -s ${model_save_path} \
        --include-package utils \
        --include-package modules \
        --file-friendly-logging \
        ${config_file}
```

Refer to `bash/train.sh` for more and detailed examples.

### Predict with the parser

The predicting command is like

```shell script
CUDA_VISIBLE_DEVICES=${gpu_id} \
    allennlp predict \
        --cuda-device 0 \
        --output-file ${output_path} \
        --predictor ${predictor_class} \
        --include-package utils \
        --include-package modules \
        --batch-size ${batch_size} \
        --silent \
        ${model_save_path} \
        ${test_set}
```

More examples in `bash/predict.sh`.

## Package structure

* `bash/` command pipelines and examples
* `config/` Jsonnet config files
* `metrics/` metrics used in training and evaluation
* `modules/` implementations of modules
* `toolkit/` external libraries and dataset tools
* `utils/` code for input/output and pre/post-processing

## Acknowledgement

Thanks to the task organizers and also thanks to the developer of AllenNLP, JAMR and TAMR.

## Contacts

For further information, please contact <lxdou@ir.hit.edu.cn>, <yxu@ir.hit.edu.cn>

[mrp-data]: http://mrp.nlpl.eu/index.php?page=4#training "mrp-data"
[mrp-sample-data]: http://svn.nlpl.eu/mrp/2019/public/sample.tgz "mrp-sample-data"
[google-drive]: https://drive.google.com/open?id=1SbtqPdNYZWY9m2cDo58tNuzCFtKUMSj1 "google-drive"
