# merge_patient_doctor_for_5a.py (簡略案)

import json

def convert(src_path, src_type, out_f):
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            hpo_id = obj.get("HPO_ID") or obj.get("hpo_id")
            HPO_name_ja = obj.get("HPO_name_ja") or obj.get("hpo_name_ja")
            if src_type == "patient":
                expr = obj.get("patient_expression_final") or obj.get("text")
            else:
                expr = obj.get("doctor_expression_final") or obj.get("text")
            if not hpo_id or not expr:
                continue

            out = {
                "hpo_id": hpo_id,
                "HPO_name_ja": obj.get("HPO_name_ja") or obj.get("hpo_name_ja"),
                "expr": expr,
                "source": src_type,  # "patient" or "doctor"
            }
            json.dump(out, out_f, ensure_ascii=False)
            out_f.write("\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient-jsonl", type=str, default="../data/LoRA_vllm/HPO_symptom_patient_expression_judge_refine.vllm_lora.jsonl")
    ap.add_argument("--doctor-jsonl", type=str, default="../data/LoRA_vllm/HPO_symptom_doctor_expression_judge_refine.vllm_lora.jsonl")
    ap.add_argument("--output-jsonl", type=str, default="../data/LoRA_vllm/merged.jsonl")
    args = ap.parse_args()

    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        convert(args.patient_jsonl, "patient", out_f)
        convert(args.doctor_jsonl, "doctor", out_f)


if __name__ == "__main__":
    main()
