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
            self.total_errors: int = 0
            self.total_words: int = 0
        else:
            print("No ground truth transcription in csv file -> no evaluation!")
            self.output_str: str = "FileName\tTranscription\n"

    def _evaluate_transcription(
        self,
        transcription_text: str,
        ground_truth_text: str,
    ) -> float:
        measures: dict = process_words(
            reference=ground_truth_text.lower(),
            hypothesis=transcription_text.lower(),
        )
        current_wer: float = measures["wer"]
        current_errors: int = (
            measures["substitutions"] + measures["deletions"] + measures["insertions"]
        )
        current_total_words: int = (
            measures["hits"] + measures["substitution"] + measures["deletions"]
        )

        self.total_errors += current_errors
        self.total_words += current_total_words

        # return wer %
        return current_wer * 100

    def _update_output(
        self,
        wav_file_path: str,
        transcription_text: str,
        ground_truth_text: str,
        wer: float,
    ):
        self.output_str += f"{os.path.basename(wav_file_path)}\t"
        self.output_str += f"{transcription_text}\t"
        if self.evaluate:
            self.output_str += f"{ground_truth_text}\t"
            self.output_str += f"{wer:.2f}\n"

    def _process_row(self, row: pd.Series) -> None:
        wav_file_path: str = row["Path"]

        print(f"Transcribing {wav_file_path}")

        result: dict = self.model.transcribe(
            wav_file_path,
            language=self.language,
        )

        transcription_text: str = result["text"]

        if self.evaluate:
            ground_truth_text: str = row["Text"]
            wer: float = self._evaluate_transcription(
                transcription_text=transcription_text,
                ground_truth_text=ground_truth_text,
            )
            print(f"Transcription of {wav_file_path} done with WER: {wer:.2f}%")

            self._update_output(
                wav_file_path=wav_file_path,
                transcription_text=transcription_text,
                ground_truth_text=ground_truth_text,
                wer=wer,
            )

        else:
            print(f"Transcription of {wav_file_path} done")
            self._update_output(
                wav_file_path=wav_file_path,
                transcription_text=transcription_text,
                ground_truth_text=None,
                wer=None,
            )

    def process_batch(self):
        for index, row in self.df.iterrows():
            self._process_row(row=row)

        # Batch stats
        if self.evaluate:
            total_wer: float = (
                (self.total_errors / self.total_words) * 100
                if self.total_words > 0
                else 0
            )
            self.output_str += f"\nTotal WER: {total_wer:.2f}%\n"
            print(f"Total WER: {total_wer:.2f}%")

        # Write output file
        with open(self.output_file, "w") as f:
            f.write(self.output_str)

        print(f"Output file ready at {self.output_file}")


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
