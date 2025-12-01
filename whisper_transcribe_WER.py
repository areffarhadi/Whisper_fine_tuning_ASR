from argparse import ArgumentParser
import os
import whisper
import pandas as pd
from jiwer import process_words


class TranscriberWER:
    def __init__(
        self,
        csv_file: str,
        output_file: str = "transcriptions_with_wer.txt",
        language: str = None,
        model: str = "large-v2",
        device: str = "cpu",
    ):
        self.model: whisper.Whisper = whisper.load_model(model, device=device)
        self.df: pd.DataFrame = pd.read_csv(csv_file)
        self.language: str = language
        self.output_file: str = output_file
        if "Text" in df.columns:
            self.evaluate: bool = True
            self.output_str: str = "FileName\tTranscription\tGroundTruth\tWER%\n"
        else:
            print("No ground truth transcription in csv file -> no evaluation!")
            self.output_str: str = "FileName\tTranscription\n"


def main() -> None:
    pass


# Load the Whisper large-v2 model on GPU
model = whisper.load_model("large-v2", device="cuda")

# Path to the CSV file
csv_file = "files_test.csv"

# Output file
output_file = "transcriptions_with_wer.txt"

# Load the CSV file
df = pd.read_csv(csv_file)

# Initialize the output file with header
with open(output_file, "w") as f:
    f.write("FileName\tTranscription\tGroundTruth\tWER%\n")

# Variables to store total errors and total number of words
total_errors = 0
total_words = 0

# Process each row in the CSV
for index, row in df.iterrows():
    wav_file_path = row["Path"]
    ground_truth_text = row["Text"]

    print(f"Transcribing {wav_file_path} ...")

    # Transcribe the audio file with language set to English
    result = model.transcribe(wav_file_path, language="en")

    # Extract the transcription text
    transcription_text = result["text"]

    # Calculate WER
    measures = process_words(
        ground_truth_text.lower(),
        transcription_text.lower(),
    )
    current_wer = measures["wer"]
    current_errors = (
        measures["substitutions"] + measures["deletions"] + measures["insertions"]
    )
    current_total_words = (
        measures["hits"] + measures["substitutions"] + measures["deletions"]
    )

    # Update total errors and words
    total_errors += current_errors
    total_words += current_total_words

    # Convert WER to percentage
    wer_percentage = current_wer * 100

    # Append the filename, transcription, ground truth, and WER to the output file
    with open(output_file, "a") as f:
        f.write(
            f"{os.path.basename(wav_file_path)}\t{transcription_text}\t{ground_truth_text}\t{wer_percentage:.2f}\n"
        )

    print(f"Transcription of {wav_file_path} done with WER: {wer_percentage:.2f}%")

# Calculate total WER
total_wer = (total_errors / total_words) * 100 if total_words > 0 else 0

# Write the total WER to the output file
with open(output_file, "a") as f:
    f.write(f"\nTotal WER: {total_wer:.2f}%\n")

print(f"Transcriptions and WERs have been saved to {output_file}")
print(f"Total WER: {total_wer:.2f}%")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "csv_file",
        type=str,
        help="Full path to the csv file containing at minimum the column 'Path' with the path to the audio files.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="transcriptions_with_wer.txt",
        help="Path to the output file (.txt). Default is './transcriptions_with_wer.txt'",
    )
    argparser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Whisper model language; if None automatic detection. Default is None",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="large-v2",
        help="Whisper model. Default is 'large-v2'",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device; default is 'cpu'",
    )
    args = argparser.parse_args()

    main(
        csv_file=args.csv_file,
        output_file=args.output_file,
        model=args.model,
        device=args.device,
    )
