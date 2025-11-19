import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import (
    set_random_seeds,
    compute_metrics,
    save_queries_and_records,
    compute_records
)
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ---------------------------
# FEW-SHOT EXAMPLES (10 shots)
# ---------------------------
FEW_SHOT = [
    ("How many flights depart from Dallas?",
     "SELECT COUNT(*) FROM flights WHERE origin_city = 'Dallas';"),

    ("List the airline names that operate flights to Chicago.",
     "SELECT DISTINCT airline_name FROM flights WHERE destination_city = 'Chicago';"),

    ("What is the maximum flight distance in the database?",
     "SELECT MAX(distance) FROM flights;"),

    ("Which flights have a distance greater than 2000 miles?",
     "SELECT flight_id FROM flights WHERE distance > 2000;"),

    ("How many unique cities are served as destinations?",
     "SELECT COUNT(DISTINCT destination_city) FROM flights;"),

    ("Show the airlines that have flights longer than 1500 miles.",
     "SELECT DISTINCT airline_name FROM flights WHERE distance > 1500;"),

    ("What is the average flight distance for flights going to Boston?",
     "SELECT AVG(distance) FROM flights WHERE destination_city = 'Boston';"),

    ("List all flights that depart from New York ordered by distance descending.",
     "SELECT flight_id, distance FROM flights WHERE origin_city = 'New York' ORDER BY distance DESC;"),

    ("Find the total number of flights operated by Delta Airlines.",
     "SELECT COUNT(*) FROM flights WHERE airline_name = 'Delta Airlines';"),

    ("Which destination city has the highest average flight distance?",
     "SELECT destination_city FROM flights "
     "GROUP BY destination_city ORDER BY AVG(distance) DESC LIMIT 1;")
]


# ---------------------------
# Argument Parser (unchanged)
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type (not used but required)')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model: gemma or codegemma')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_name', type=str, default='experiment')
    return parser.parse_args()


# ---------------------------
# PROMPT CREATION
# ---------------------------
def create_prompt(sentence, k):
    """
    Create zero-shot or k-shot prompt using fixed FEW_SHOT examples.
    """

    prompt = (
        "You are an expert SQL generation system for a flight booking database.\n"
        "Your task is to convert a natural language question into a **valid SQL query**.\n"
        "Rules:\n"
        "- Return ONLY the SQL query.\n"
        "- The query must end with a semicolon.\n"
        "- Use only the database schema.\n"
        "- Do NOT output explanations.\n\n"
        "### Examples\n"
    )

    # add k examples
    for i in range(min(k, len(FEW_SHOT))):
        nl, sql = FEW_SHOT[i]
        prompt += f"Q: {nl}\nSQL: {sql}\n\n"

    # add the real question
    prompt += f"### Now answer:\nQ: {sentence}\nSQL:"
    return prompt


# ---------------------------
# EXPERIMENT: K-SHOT
# ---------------------------
def exp_kshot(tokenizer, model, inputs, k):
    raw_outputs, extracted_queries = [], []
    MAX_NEW_TOKENS = 128

    for sentence in tqdm(inputs):
        prompt = create_prompt(sentence, k)
        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_outputs.append(response)

        sql = extract_sql_query(response)
        extracted_queries.append(sql)

    return raw_outputs, extracted_queries


# ---------------------------
# EVALUATION
# ---------------------------
def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )

    error_rate = sum(1 for e in model_error_msgs if e) / max(1, len(model_error_msgs))

    return sql_em, record_em, record_f1, model_error_msgs, error_rate


# ---------------------------
# MODEL INITIALIZATION
# ---------------------------
def initialize_model_and_tokenizer(model_name, to_quantize=False):

    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        model = GemmaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(DEVICE)

    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)

        if to_quantize:
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config
            ).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            ).to(DEVICE)

    return tokenizer, model


# ---------------------------
# MAIN
# ---------------------------
def main():
    args = get_args()
    set_random_seeds(args.seed)

    shot = args.shot
    experiment_name = args.experiment_name
    model_name = args.model

    # load NL/SQL
    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    # load model
    tokenizer, model = initialize_model_and_tokenizer(model_name, args.quantization)

    for split in ["dev", "test"]:
        eval_x = dev_x if split == "dev" else test_x
        gt_sql_pth = f"data/{split}.sql"
        gt_record_path = f"records/{split}_gt_records.pkl"

        # run prompting
        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot)

        # save generated SQL
        model_sql_path = f"results/{experiment_name}_{split}.sql"
        model_record_path = f"records/{experiment_name}_{split}.pkl"

        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)

        # evaluate
        sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
            eval_x, None, gt_sql_pth, model_sql_path, gt_record_path, model_record_path
        )

        print(f"\n===== {split.upper()} RESULTS =====")
        print(f"SQL EM: {sql_em:.4f}")
        print(f"Record EM: {record_em:.4f}")
        print(f"Record F1: {record_f1:.4f}")
        print(f"SQL Error Rate: {error_rate:.2%}")

        save_logs(
            f"logs/{experiment_name}_{split}.txt",
            sql_em, record_em, record_f1, model_error_msgs
        )


if __name__ == "__main__":
    main()
