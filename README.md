# Whisper fine-tuning for ASR
In this Repo, you can easily fine-tune different variations of the Whisper model to your specific multilingual data based on a simple manifest. 

1. [prepare_data.py](prepare_data.py)     :::: to prepare ".csv" files for train and test
2. [train.py](train.py)                   :::: train and save the fine-tuned Whisper model
3. [decode.py](decode.py)                 :::: decode the test or any evaluation ".wav" file


## Auxilary files

The required packages are listed in the "requirements.txt" file and you can easily install all of them using: 
`pip install -r requirements.txt`

It would be better to make a new Python environment using `python3 -m venv myenv` , after that, activate the venv using `source myenv/bin/activate` and then install the packages.

To run on the servers by Slurm, you can use the [slurm_run.sh](slurm_run.sh) file.

The "files_test.csv" and "files_train.csv" help us understand better the required files for testing and training.
