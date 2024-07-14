from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio
import pandas as pd
import gc
import evaluate
import torch
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load the training and test data
train_df = pd.read_csv("./files_train.csv")
test_df = pd.read_csv("./files_test.csv")

# Convert the pandas dataframes to dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Convert the sample rate of every audio file using cast_column function
train_dataset = train_dataset.cast_column("Path", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("Path", Audio(sampling_rate=16000))

# Load WhisperTokenizer and WhisperProcessor
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="English", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

# Define a function to prepare the dataset
def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    audio = examples["Path"]
    examples["input_features"] = feature_extractor(
        audio["array"], sampling_rate=16000).input_features[0]
    del examples["Path"]
    sentences = examples["Text"]

    # encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["Text"]
    return examples


# Prepare the train and test datasets
train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

# Define the data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Define the compute_metrics function
# wer = evaluate.load("wer")
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output_dir",  # change to a repo name of your choice
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=40000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10000,
    eval_steps=5000,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# Initialize the model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("step1 - training...")
train_result = trainer.train()

# Save the model, tokenizer, and processor
trainer.save_model()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

print("step2 - ")

metrics = train_result.metrics
print("step3")
max_train_samples = len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))
print("step4 - saving the model...")
trainer.save_model()
print("model created!")
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()




# Evaluate the model
print("Evaluating the final trained model...")
metrics = trainer.evaluate()
print("Evaluation complete. Metrics:", metrics)
