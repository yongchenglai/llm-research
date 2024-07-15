# bert_quick_demo.py
# bert_quick_demo.py --model_name_or_path="google-bert/bert-base-uncased"
from transformers import BertTokenizer, BertModel
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "google-bert/bert-base-uncased"
    parser.add_argument("--model_name_or_path", type=str, help='mode name or path')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path)
    print(model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)


