- [VTG-GPT](#vtg-gpt)
  - [Preparation](#preparation)
  - [Inference on QVHighlights val split](#inference-on-qvhighlights-val-split)
  - [MiniGPT-v2 for Image captioning](#minigpt-v2-for-image-captioning)
  - [Baichuan2 for Query debiasing](#baichuan2-for-query-debiasing)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


# VTG-GPT

This is our implementation for the paper **VTG-GPT: Tuning-Free Zero-Shot Video Temporal Grounding with GPT**.

![Alt text](manuscript/pipeline.png)

## Preparation

1. Install dependencies

```sh
conda create -n vtg-gpt python=3.10
conda activate vtg-gpt
pip install -r requirements.txt
```

2. Unzip caption files

```sh
cd data/qvhighlights/caption/
unzip test.zip
unzip val.zip
```


## Inference on QVHighlights val split

```sh
# inference
python infer_qvhighlights.py --val

# evaluation
bash standalone_eval/eval.sh
```

Run the above code to get:

| Metrics| R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP@avg |
| -----  | ------ | ------ | ------- | -------- | ------- |
| Values | 59.03  | 38.90   | 56.11   | 35.44    | 35.57   |


## MiniGPT-v2 for Image captioning
TODO

## Baichuan2 for Query debiasing
TODO

# Acknowledgement

[Moment-DETR](https://github.com/jayleicn/moment_detr)
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
[Baichuan2](https://github.com/baichuan-inc/Baichuan2)


# Citation
If you find this project useful for your research, please kindly cite our paper.
```
@article{xu2024vtg,
  title={VTG-GPT: Tuning-Free Zero-Shot Video Temporal Grounding with GPT},
  author={Xu, Yifang and Sun, Yunzhuo and Xie, Zien and Zhai, Benxiang and Du, Sidan},
  journal={Applied Sciences},
  volume={14},
  number={5},
  pages={1894},
  year={2024},
  publisher={MDPI}
}
```
