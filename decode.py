from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio
import pandas as pd
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import re

# Load the test data
test_df = pd.read_csv("./files_test.csv")
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.cast_column("Path", Audio(sampling_rate=16000))

# Function to prepare dataset
def prepare_dataset(examples):
    # Compute log-Mel input features from input audio array
    audio = examples["Path"]
    examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    del examples["Path"]
    sentences = examples["Text"]

    # Encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["Text"]
    return examples 

#### lines 32-35 for utilizing pre-trained model  ####
#### lines 42-45 for utilizing fine-tuned model  ####


##### pre-trained - Load the  model, feature extractor, tokenizer, and processor with the specified language and task (https://huggingface.co/openai/whisper-large)
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="en", task="transcribe", pad_token="<pad>")
# processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="en", task="transcribe")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")

# directory of fine-tuned model
output_dir = "./output_dir"

#### fine-tuned - Load the  model, feature extractor, tokenizer, and processor with the specified language and task 
feature_extractor = WhisperFeatureExtractor.from_pretrained(output_dir)
tokenizer = WhisperTokenizer.from_pretrained(output_dir, language="en", task="transcribe", pad_token="<pad>")
processor = WhisperProcessor.from_pretrained(output_dir, language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(output_dir)


# Debugging print statements to confirm language setting
print("Tokenizer language:", tokenizer.language)
print("Processor language:", processor.tokenizer.language)

# Prepare the test dataset
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output_dir",  # change to a repo name of your choice
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=1,  
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=15000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5000,
    eval_steps=5000,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Define the data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # If bos token is appended in previous tokenization step, cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        # Create attention mask for input features
        batch["attention_mask"] = torch.ones(batch["input_features"].shape[:2], dtype=torch.long)
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

# Function to clean text (remove Punctuations and convert to uppercase)
def clean_text(text):
    text = text.upper()  # Convert to uppercase
    text = text.replace('. , ! ?', '')  # Remove dots
    return text

# Compute metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode the predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Clean the text (uppercase and remove dots)
    pred_str = [clean_text(pred) for pred in pred_str]
    label_str = [clean_text(label) for label in label_str]

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
    # Save predictions and ground truth to a text file
    output_file = "predictions_and_ground_truth.txt"
    with open(output_file, "w") as f:
        for pred, label in zip(pred_str, label_str):
            f.write(f"Predicted: {pred}\n")
            f.write(f"Grnd Trth: {label}\n")
            f.write("\n")
    
    return {"wer": wer}

# Initialize the trainer with the loaded model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Evaluate the model
print("Evaluating the final trained model...")
metrics = trainer.evaluate()

# Log and save the evaluation metrics
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
print("Evaluation complete. Metrics:", metrics)
print(f"Predictions and ground truth saved to {output_file}")
