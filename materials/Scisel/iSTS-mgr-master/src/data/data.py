from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import torch
import os
import fnmatch
from src.data.text import clear_alignment, clear_sentence, source_to_words
from src.constants import TRAIN_DATA_PATH, TEST_DATA_PATH, DATASETS

def prepare_data(path):

    directory, file = os.path.split(path)
    filename, _ = os.path.splitext(file)

    path_processed = f'{directory}/processed_{filename}.xml'

    process_dataset(path, path_processed)

    return path_processed


def process_dataset(path, path_processed):
    with open(path, 'rb') as reader:
        with open(path_processed, 'wb') as writer:
            text = reader.read()
            text = text.replace(b'<==>', b'---')
            text = text.replace(b'&', b'und')
            writer.write(text)

    start_root = '<root>'
    end_root = '</root>'

    with open(path_processed, "a+") as f:
        f.write(end_root)

    with open(path_processed, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(start_root + '\n' + content)


def df_from_alignment(series):
    new_df = pd.DataFrame(columns=[
        'class_1', 'class_2', 'chunks_source', 'chunks_translation', 'text_source', 'text_translation'
    ])

    rows = []
    for data in series:
        for element in data:
            sentence = element[4]
            row = {
                'class_1': element[0],
                'class_2': element[1],
                'chunks_source': element[2],
                'chunks_translation': element[3],
                'text_source': sentence[0],
                'text_translation': sentence[1],
            }
            rows.append(row)
            # new_df = new_df.append(row, ignore_index=True)
    new_df = pd.DataFrame(rows)
    return new_df


def get_classes_weight(df_labels):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_labels), y=np.array(df_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list, torch.LongTensor(label_list)


def get_df_from_wa(path):
    df = pd.read_xml(path)
    df.drop('status', axis=1, inplace=True)

    df['source_split'] = df['source'].apply(source_to_words)
    df['translation_split'] = df['translation'].apply(source_to_words)

    split_columns = df['sentence'].str.split(pat='\n', expand=True)
    df['source'] = split_columns[0]
    df['translation'] = split_columns[1]
    df.drop('sentence', axis=1, inplace=True)

    df['source'] = df['source'].apply(clear_sentence)
    df['translation'] = df['translation'].apply(clear_sentence)
    df.dropna(how='any', axis=0, inplace=True)
    df = df[df['alignment'].str.strip() != '']
    df['alignment'] = df['alignment'].apply(clear_alignment)

    new_df = df_from_alignment(df['alignment'])

    def change_alignment(text):
        class_1_change = {
            'SPE1_FACT': 'SPE1',
            'SIMI_FACT': 'SIMI',
            'EQUI_POL': 'EQUI',
            'EQUI_FACT': 'EQUI',
            'REL_POL': 'REL',
            'SPE2_FACT': 'SPE2',
            'NOALI_FACT': 'NOALI',
            'SPE2_POL': 'SPE2'
        }
        try:
            new_class_name = class_1_change[text]
            return new_class_name
        except KeyError as e:
            return text

    def change_score(text):
        class_2_change = {
            "NIL": "0"
        }
        try:
            new_score = class_2_change[text]
            return new_score
        except KeyError as e:
            return text

    new_df['alignment'] = new_df['class_1'].apply(change_alignment)
    new_df['score'] = new_df['class_2'].apply(change_score)
    # df_labels_1 = new_df.pop('class_1')
    # df_labels_2 = new_df.pop('class_2')

    new_df.drop(columns=['chunks_source', 'chunks_translation', 'class_2', 'class_1'], inplace=True)

    # if labels == 1:
    #     df_labels = df_labels_1
    # else:
    #     df_labels = df_labels_2
    # return new_df, df_labels
    return new_df

def read_raw_file(filename):

    with open(filename) as f:
        texts = []    
        for line in f:
            texts.append(line.strip())
    
    return texts

def find_files_with_phrase(directory, phrase, extension_to_ommit="xml"):
    matching_files = []
    # Traverse the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the phrase is in the filename
            if fnmatch.fnmatch(file, f'*{phrase}*') and not fnmatch.fnmatch(file, f'*.{extension_to_ommit}'):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def get_dataframe(dataset_name, subset, source):
    if subset == "test":
        data_dir = TEST_DATA_PATH
    elif subset == "train":
        data_dir = TRAIN_DATA_PATH
    files = find_files_with_phrase(data_dir / source, dataset_name)
    assert len(files) < 3 or len(files) == 0, "Wrong files"
    assert dataset_name in DATASETS, "No such dataset available, wrong name"
    df = pd.DataFrame()
    
    if source in ["raw", "chunked"]:
        for i, file in enumerate(files):
            df[f'sent{i+1}'] = read_raw_file(file)
    else:
        prepare_data_path = prepare_data(files[0])
        df = get_df_from_wa(prepare_data_path)
        df["merged"] = df["score"].astype(str) + "-" + df["alignment"].astype(str)
        df["both"] = df["merged"]
        
    return df
