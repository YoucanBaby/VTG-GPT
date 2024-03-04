#!/usr/bin/env bash
# Usage: bash standalone_eval/eval.sh

data_split=val

submission_path=outputs/qvhighlights/infer_${data_split}.jsonl
gt_path=data/qvhighlights/gt/highlight_${data_split}_release.jsonl
save_path=outputs/qvhighlights/infer_${data_split}_metrics.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
