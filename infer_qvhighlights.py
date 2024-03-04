import argparse
import os
from typing import List

from tqdm import tqdm
import numpy as np
import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer

from standalone_eval.file_utils import load_jsonl, save_jsonl


class VTG_GPT:
    def __init__(self, num_bins=10, top_k=8, gap=6, nms_threshold=0.95):
        self.num_bins = num_bins
        self.top_k = top_k
        self.gap = gap
        self.nms_threshold = nms_threshold
        self.similarity_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        
    @torch.no_grad()
    def locate_span(self, qid, vid, query, caption_list, rephrased_query_list, gt_span_list=None):
        normalized_scores = self.get_normalized_scores(query, caption_list)
        span_list = self.get_span(
            normalized_scores,
            num_bins=self.num_bins,
            top_k=self.top_k,
            gap=self.gap,
        )
        
        for rephrased_query in rephrased_query_list:
            rephrased_normalized_scores = self.get_normalized_scores(rephrased_query, caption_list)
            normalized_scores += rephrased_normalized_scores
            
            rephrased_span_list = self.get_span(
                rephrased_normalized_scores,
                num_bins=self.num_bins,
                top_k=self.top_k,
                gap=self.gap,
            )
            span_list.extend(rephrased_span_list)
        span_list.sort(key=lambda x: x[2], reverse=True)
        
        normalized_scores /= len(rephrased_query_list) + 1
        
        if self.nms_threshold > 0:
            span_list = self.nms(span_list, self.nms_threshold)
        
        res = {
            "qid": qid,
            "query": query,
            "vid": vid,
            "relevant_windows": gt_span_list,
            "pred_relevant_windows": [[s[0]*2, (s[1]+1)*2, s[2]] for s in span_list], 
            "span_index_list": span_list,
            "pred_saliency_scores": normalized_scores.tolist(),
        }
        return res

    def get_normalized_scores(self, query: str, caption_list: List[str]):
        embed_query = self.similarity_model.encode(query, convert_to_tensor=True)
        embed_caption_list = self.similarity_model.encode(caption_list, convert_to_tensor=True)

        cos_value = sentence_transformers.util.pytorch_cos_sim(embed_query, embed_caption_list)[0]
        cos_value = cos_value.cpu().numpy()

        def normalize(value): return (value - value.min()) / (value.max() - value.min())
        normalized_scores = normalize(cos_value)
        return normalized_scores
    
    def get_span(self, scores: np.ndarray, num_bins: int, top_k: int, gap: int):
        # compute histogram, dividing the range into 10 equal parts
        counts, thresholds = np.histogram(scores, bins=num_bins, range=(0, 1))

        # get dynamic threshold
        threshold = 0
        for i in range(len(counts)-1, -1, -1):
            total_num = sum(counts[i:])
            if total_num >= top_k:
                threshold = thresholds[i]
                break

        top_k_moments = np.where(scores > threshold)[0]

        proposal_list = [[top_k_moments[0]]]
        for moment in top_k_moments[1:]:
            if moment - proposal_list[-1][-1] <= gap:
                proposal_list[-1].append(moment)
            else:
                proposal_list.append([moment])
        
        # TODO optimize span_scores
        all_len = sum([len(p) for p in proposal_list])
        len_scores = [len(p) / all_len for p in proposal_list]
        
        proposal_scores = [np.mean(scores[p]) for p in proposal_list]
        len_weight = 0.5
        score_weight = 0.5
        span_scores = [len_weight * len_scores[i] + proposal_scores[i] * score_weight 
                       for i in range(len(proposal_list))]
        
        span_list = []
        for i in range(len(proposal_list)):
            span_list.append([min(proposal_list[i]), max(proposal_list[i]), span_scores[i]])
        span_list.sort(key=lambda x: x[2], reverse=True)
        return span_list

    def iou(self, span1, span2):
        start1, end1, _ = span1
        start2, end2, _ = span2

        # compute intersection
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        inter = max(0, inter_end - inter_start)

        # compute union
        union = (end1 - start1) + (end2 - start2) - inter
        return inter / union if union != 0 else 0

    def nms(self, span_list, iou_threshold):
        # Sort by score in descending order
        span_list = sorted(span_list, key=lambda x: x[2], reverse=True) 
        keep = []

        while span_list:
            highest_score_span = span_list.pop(0)
            keep.append(highest_score_span)
            span_list = [span for span in span_list if self.iou(highest_score_span, span) < iou_threshold]

        return keep


def main(args):
    # load model
    model = VTG_GPT(
        num_bins=args.num_bins,
        top_k=args.top_k,
        gap=args.gap,
        nms_threshold=0.95,
    )

    # dir & path
    caption_dir = f"data/qvhighlights/caption/{args.data_split}"
    annotation_path = f"data/qvhighlights/query/{args.data_split}.jsonl"
    output_path = f"outputs/qvhighlights/infer_{args.data_split}.jsonl"

    # load files
    annotation = load_jsonl(annotation_path)
    res_list = []
    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for i, item in tqdm(enumerate(annotation), desc="Processing"):
        qid = item["qid"]
        vid = item["vid"]
        query = item["query"]
        gt_span_list = item["relevant_windows"] if "relevant_windows" in item else None
        rephrased_query_list = item["rephrased_query"]
        
        caption = load_jsonl(os.path.join(caption_dir, f"{vid}.jsonl"))
        caption_list = [c["description"] for c in caption]
        
        res_dict = model.locate_span(qid, vid, query, caption_list, rephrased_query_list, gt_span_list)
        res_list.append(res_dict)

        if i == 5 and args.debug:
            save_jsonl(res_list, output_path.replace("infer", "debug"))
            return

    save_jsonl(res_list, output_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for qvhighlights dataset using VTG-GPT.")

    parser.add_argument("data_split", choices=["train", "val", "test"], help="qvhighlights dataset split: train, val, or test")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--num_bins', default=10, type=int, help='Number of histogram bins')
    parser.add_argument('--top_k', default=8, type=int, help='Use top k moments to compute dynamic threshold')
    parser.add_argument('--gap', default=6, type=int, help='Maximum gap between two moments in a span')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
