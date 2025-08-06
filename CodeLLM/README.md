# Experiments based on CodeT5+

We performed all experiments on the models which W.

# How to Use?

## Environment
Download the test datasets from https://drive.google.com/file/d/1xw6Edqf_nknKoei_LC49n4EtvNQezKGe/view and move to CodeTransOcean/CodeLLM/data.

```bash
cd CodeTransOcean/CodeLLM
python -m venv env
source env/bin/activate
pip3 install torch torchvision torchaudio # It depends on your environment.
pip install -r requirements.txt
```

## Finetuning & Inference & Evaluation

``` run_preprocess.py ``` is used to pre-process data.

``` run_translation.py ``` is used for training and inference on a specified dataset.

``` run_score.py ```  and ``` evaluator ``` are used to calculate inference results in the BLEU score.

Other ```.sh``` files are used to specify which multilingual modeling methods to use on which datasets to train CodeT5+ and infer.


## Experimental results

<div align="center">
  <img src="./images/MultilingualTrans.png">
  <img src="./images/NicheTrans.png">
  <img src="./images/DLTrans.png">
</div>

For more detailed experimental results, please see our paper.
