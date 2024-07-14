import os
import pandas as pd

# Define the path to your .wav files directory and the .txt file
wav_dir = '/home/arfarh/data/Aref_tools/openai/WTIMIT_data_safe_folder/dev_clean_2_W_CH'
txt_file = '/home/arfarh/data/Aref_tools/openai/WTIMIT_data_safe_folder/dev_clean_2_W_CH/text'
output_csv = 'files_test_WSD_CH.csv'



# Define your custom set of punctuation characters
custom_punctuation = '. , ! ?'


# Read the .txt file
file_to_text = {}
with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            filename, text = parts
            file_to_text[filename] = text

# Recursively search for .wav files
wav_paths = []
for root, _, files in os.walk(wav_dir):
    for file in files:
        if file.endswith('.wav'):
            wav_paths.append(os.path.join(root, file))

# Function to remove specified punctuation
def remove_punctuation(text, punctuation):
    return text.translate(str.maketrans('', '', punctuation))

# Prepare data for the CSV
data = []
for wav_path in wav_paths:
    wav_filename = os.path.basename(wav_path)
    filename_without_ext = os.path.splitext(wav_filename)[0]  # Remove the .wav extension
#    lookup_key = filename_without_ext[4:9]  # Extract characters 4 to 8 (0-based index)
#    text = file_to_text.get(lookup_key, '').strip().upper()  # Lookup the corresponding text
    text = file_to_text.get(filename_without_ext, '').strip().upper()
    text = remove_punctuation(text, custom_punctuation)  # Remove custom punctuation from the text
    data.append([wav_path, text])

# Create a DataFrame and write to CSV
df = pd.DataFrame(data, columns=['Path', 'Text'])
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f'CSV file has been created at {output_csv}')
