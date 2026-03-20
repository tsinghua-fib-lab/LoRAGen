# LoRAGen/stage2/train/evaluation/dataset_utils.py
from typing import List
import os
import json

# Mapping from task short-name to JSON file name
NAME_MAPPING = {
    "glue_stsb": "glue_stsb:2.0.0.json",
    "gem_e2e_nlg": "gem_e2e_nlg:1.1.0.json",
    "super_glue_wic": "super_glue_wic:1.0.2.json",
    "glue_sst2": "glue_sst2:2.0.0.json",
    "newsroom": "newsroom:1.0.0.json",
    "glue_wnli": "glue_wnli:2.0.0.json",
    "yelp_polarity_reviews": "yelp_polarity_reviews:0.2.0.json",
    "glue_qnli": "glue_qnli:2.0.0.json",
    "paws_wiki": "paws_wiki:1.1.0.json",
    "super_glue_wsc.fixed": "super_glue_wsc.fixed:1.0.2.json",
    "glue_mrpc": "glue_mrpc:2.0.0.json",
    "quac": "quac:1.0.0.json",
    "gem_web_nlg_en": "gem_web_nlg_en:1.1.0.json",
    "anli_r1": "anli_r1:0.1.0.json",
    "anli_r2": "anli_r2:0.1.0.json",
    "anli_r3": "anli_r3:0.1.0.json",
    "glue_cola": "glue_cola:2.0.0.json",
    "ag_news_subset": "ag_news_subset:1.0.0.json",
    "ai2_arc_ARC-Challenge": "ai2_arc_ARC-Challenge:1.0.0.json",
    "ai2_arc_ARC-Easy": "ai2_arc_ARC-Easy:1.0.0.json",
    "cnn_dailymail": "cnn_dailymail:3.4.0.json",
    "coqa": "coqa:1.0.0.json",
    "cosmos_qa": "cosmos_qa:1.0.0.json",
    "drop": "drop:2.0.0.json",
    "gigaword": "gigaword:1.2.0.json",
    "glue_mnli": "glue_mnli:2.0.0.json",
    "glue_qqp": "glue_qqp:2.0.0.json",
    "hellaswag": "hellaswag:1.1.0.json",
    "huggingface:xsum": "huggingface:xsum.json",
    "imdb_reviews_plain_text": "imdb_reviews_plain_text:1.0.0.json",
    "kilt_tasks_hotpotqa_combining_facts": "kilt_tasks_hotpotqa_combining_facts.json",
    "kilt_tasks_hotpotqa_complex_question": "kilt_tasks_hotpotqa_complex_question.json",
    "kilt_tasks_hotpotqa_final_exam": "kilt_tasks_hotpotqa_final_exam.json",
    "kilt_tasks_hotpotqa_formulate": "kilt_tasks_hotpotqa_formulate.json",
    "kilt_tasks_hotpotqa_straighforward_qa": "kilt_tasks_hotpotqa_straighforward_qa.json",
    "lambada": "lambada:1.0.0.json",
    "multi_news": "multi_news:1.0.0.json",
    "natural_questions_open": "natural_questions_open:1.0.0.json",
    "openbookqa": "openbookqa:0.1.0.json",
    "para_crawl_enes": "para_crawl_enes.json",
    "piqa": "piqa:1.0.0.json",
    "qasc_is_correct_1": "qasc_is_correct_1.json",
    "qasc_is_correct_2": "qasc_is_correct_2.json",
    "qasc_qa_with_combined_facts_1": "qasc_qa_with_combined_facts_1.json",
    "qasc_qa_with_separated_facts_1": "qasc_qa_with_separated_facts_1.json",
    "qasc_qa_with_separated_facts_2": "qasc_qa_with_separated_facts_2.json",
    "qasc_qa_with_separated_facts_3": "qasc_qa_with_separated_facts_3.json",
    "qasc_qa_with_separated_facts_4": "qasc_qa_with_separated_facts_4.json",
    "qasc_qa_with_separated_facts_5": "qasc_qa_with_separated_facts_5.json",
    "quail_no_prompt_text": "quail_no_prompt_text.json",
    "quartz_answer_question_below": "quartz_answer_question_below.json",
    "quartz_read_passage_below_choose": "quartz_read_passage_below_choose.json",
    "quoref_Given_Context_Answer_Question": "quoref_Given_Context_Answer_Question.json",
    "race_high_Select_the_best_answer": "race_high_Select_the_best_answer.json",
    "race_high_Taking_a_test": "race_high_Taking_a_test.json",
    "race_middle_Select_the_best_answer": "race_middle_Select_the_best_answer.json",
    "race_middle_Taking_a_test": "race_middle_Taking_a_test.json",
    "ropes_prompt_mix": "ropes_prompt_mix.json",
    "sciq_Direct_Question": "sciq_Direct_Question.json",
    "sciq_Multiple_Choice": "sciq_Multiple_Choice.json",
    "samsum": "samsum:1.0.0.json",
    "snli": "snli:1.1.0.json",
    "social_i_qa_Generate_answer": "social_i_qa_Generate_answer.json",
    "squad_v1.1": "squad_v1.1:3.0.0.json",
    "squad_v2.0": "squad_v2.0:3.0.0.json",
    "story_cloze_2016": "story_cloze_2016:1.0.0.json",
    "super_glue_cb": "super_glue_cb:1.0.2.json",
    "super_glue_copa": "super_glue_copa:1.0.2.json",
    "super_glue_multirc": "super_glue_multirc:1.0.2.json",
    "super_glue_record": "super_glue_record:1.0.2.json",
    "super_glue_rte": "super_glue_rte:1.0.2.json",
    "trec": "trec:1.0.0.json",
    "trivia_qa_rc": "trivia_qa_rc:1.1.0.json",
    "web_questions_question_answer": "web_questions_question_answer.json",
    "wiki_bio_who": "wiki_bio_who.json",
    "wiqa_effect_with_label_answer": "wiqa_effect_with_label_answer.json",
    "winogrande": "winogrande:1.1.0.json",
    "wmt16_translate_de-en": "wmt16_translate_de-en:1.0.0.json",
    "wmt16_translate_ru-en": "wmt16_translate_ru-en:1.0.0.json",
}

def get_examples_for_inference(task_names: List[str]):
    examples = []

    def extract_short_name(task_name: str):
        return task_name.split("lorahub_flan_t5_large-")[-1]

    data_root = os.environ.get("LORAGEN_DATA_ROOT", "./datasets/flanv2")

    for task_name in task_names:
        short_name = extract_short_name(task_name)
        json_file_name = NAME_MAPPING.get(short_name, f"{short_name}.json")
        file_path = os.path.join(data_root, json_file_name)
        prefix = ""  # optional prompt prefix

        if not os.path.exists(file_path):
            print(f"[WARN] Dataset not found: {file_path}")
            continue

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[JSON error] {e}")
                    continue

                inputs = str(item.get("inputs", "")).strip()
                targets = str(item.get("targets", "")).strip()
                if not inputs or not targets:
                    continue

                examples.append({
                    "input": f"{prefix}{inputs}\nAnswer:",
                    "output": targets,
                    "task_name": task_name
                })

    return examples
