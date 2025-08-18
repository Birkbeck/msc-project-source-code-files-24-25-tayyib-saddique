import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

class WorkerPool:
    """
    WorkerPool loads models once per GPU device and processes batches of files on that GPU.
    """
    party_model_name = "m-newhauser/distilbert-political-tweets"
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, device_id, batch_size=512):
        self.device = torch.device(f"cuda:{device_id}")
        self.batch_size = batch_size
        torch.cuda.set_device(self.device)

        # Load party classification model/tokenizer
        self.party_tokenizer = AutoTokenizer.from_pretrained(self.party_model_name)
        self.party_model = AutoModelForSequenceClassification.from_pretrained(self.party_model_name).to(self.device)
        self.party_model.eval()

        # Load sentiment classification model/tokenizer
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name).to(self.device)
        self.sentiment_model.eval()

    @staticmethod
    def classify_texts(texts, tokenizer, model, device, batch_size=512, threshold=None):
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        results = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(device)
                batch_attention_mask = attention_mask[i:i+batch_size].to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    scores, preds = torch.max(probs, dim=1)

                for score, pred in zip(scores.cpu(), preds.cpu()):
                    label = model.config.id2label[pred.item()].lower()
                    if threshold is not None and score.item() < threshold:
                        label = "unknown"
                    results.append(label)

        return results

    def process_single_file(self, file_path):
        print(f"[GPU {self.device}] Processing {file_path}...")
        df = pd.read_csv(
            file_path,
            compression="gzip",
            usecols=[
                "id", "rawContent", "lang", "date", "replyCount", "retweetCount",
                "likeCount", "quoteCount", "hashtags", "viewCount"
            ]
        )
        df = df[df["lang"] == "en"]
        if df.empty:
            print(f"[GPU {self.device}] No English tweets in file, skipping.")
            return None

        texts = df['rawContent'].tolist()

        # Classify political party
        start = time.time()
        df['party'] = self.classify_texts(
            texts, self.party_tokenizer, self.party_model, self.device,
            self.batch_size, threshold=0.6
        )
        print(f"[GPU {self.device}] Party classification done in {time.time() - start:.2f}s")

        # Classify sentiment
        start = time.time()
        df['sentiment'] = self.classify_texts(
            texts, self.sentiment_tokenizer, self.sentiment_model, self.device, self.batch_size
        )
        print(f"[GPU {self.device}] Sentiment classification done in {time.time() - start:.2f}s")

        output_dir = os.path.join(os.getcwd(), "x_processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, os.path.basename(file_path).replace(".csv.gz", "_processed.parquet")
        )
        df.to_parquet(output_path, index=False, engine="pyarrow")

        return output_path

    def process_files(self, file_list):
        results = []
        for file_path in file_list:
            res = self.process_single_file(file_path)
            if res:
                results.append(res)
        return results

    @classmethod
    def process_files_on_device(cls, file_list, device_id):
        worker = cls(device_id)
        return worker.process_files(file_list)
