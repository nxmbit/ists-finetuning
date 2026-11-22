import re
import os
import pandas as pd

# Generator dla klasy 1
def gen_1(preds_1):
    for i in preds_1:
        yield i

# Generator dla klasy 2
def gen_2(preds_2):
    for i in preds_2:
        yield i


def to_correct_class(idx, class_type):
    type_1 = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
    type_1 = {
        'EQUI': 'EQUI', 
        'NOALI': 'NOALI', 
        'OPPO': 'OPPO', 
        'REL': 'REL', 
        'SIMI': 'SIMI', 
        'SPE1': 'SPE1', 
        'SPE2': 'SPE2'
    }
    type_2 = ['NIL', '1', '2', '3', '4', '5']
    if class_type == 1:
        return type_1[idx]
    else:
        return type_2[idx]


def modify_alignment(text, g_1, g_2, filename):
    chunks = text.split('\n')[1:-1]
    new_chunks = ['']
    for chunk in chunks:
        # Np. do NIL, 0, 1, ..., 5
        score = to_correct_class(next(g_2), 2)
        align = to_correct_class(next(g_1), 1)
        
        # chunk = re.sub(r'\s\/\/\s[A-Z]+[1-2]*\s\/\/\s', f' // {align} // ', chunk)
        # chunk = re.sub(r'\s\/\/\s(\bNIL\b)*[0-6]*\s\/\/\s', f' // {score} // ', chunk)
        # Np. do EQUI, SPE1, SPE2, ...
        prefix, _, _, rest = chunk.split(" // ", 3)
        chunk = f"{prefix} // {align} // {score} // {rest}"

        # # Jak NIL to musi być NOALI
        # chunk = re.sub(r'\s\/\/\s(.*?)\s\/\/\s(\bNIL\b)\s\/\/\s', f' // NOALI // NIL // ', chunk)
        # # # A Jak NOALI to musi być 0
        # chunk = re.sub(r'\s\/\/\s(\bNOALI\b)\s\/\/\s(.*?)\s\/\/\s', f' // NOALI // NIL // ', chunk)
        chunk = re.sub(r'\s\/\/\s(.*?)\s\/\/\s(\bNIL\b)\s\/\/\s', f' // NOALI // 0 // ', chunk)
        # A Jak NOALI to musi być 0
        chunk = re.sub(r'\s\/\/\s(\bNOALI\b)\s\/\/\s(.*?)\s\/\/\s', f' // NOALI // 0 // ', chunk)
        new_chunks.append(chunk)
    new_chunks.append('')
    text = '\n'.join(new_chunks)

    return text


def get_rows_with_nan(df):
    rows_with_nan = []
    for index, row in df.iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)
    return rows_with_nan


def prepare_test_wa_file(test_path, alignment_pred, score_pred, filename):
    g_alignment = gen_1(alignment_pred)
    g_score = gen_2(score_pred)

    df = pd.read_xml(test_path)
    df.drop('status', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    rows_with_nan = get_rows_with_nan(df)
    rows_with_nan = [x+1 for x in rows_with_nan]
    df.dropna(axis=0, how='any', inplace=True)
    df['alignment'] = df['alignment'].apply(modify_alignment, args=(g_alignment, g_score, filename))
    df.to_xml(filename, index=False, row_name='Sentence', root_name='root', xml_declaration=False, pretty_print=True)
    text = ''
    tags = ['root', 'sentence', 'source', 'translation', 'alignment']
    with open(filename, 'r') as file:
        text = file.read()
        text = text.replace('<root>', '')
        text = text.replace('</root>', '')
        text = text.replace('<sentence>', '')
        text = text.replace('</sentence>', '')
        text = text.replace('<Sentence>', '<sentence>')
        text = text.replace('</Sentence>', '</sentence>')
        text = text.replace('---', '<==>')
        text = text.replace('und', '&')
        for tag in tags:
            text = text.replace(f'<{tag}>', f'\n<{tag}>\n')
            text = text.replace(f'</{tag}>', f'\n</{tag}>\n')
        text = text.strip()
        text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    with open(filename, 'w') as file:
        file.write(text)

    with open(filename, 'r') as file:
        lines = file.readlines()
        index = 1
        new_lines = []
        for line in lines:
            if '<sentence>' in line:
                if index in rows_with_nan:
                    index += 1
                line = line.replace('<sentence>', f'<sentence id=\"{index}\" status=\"\">')
                index += 1
            if '</sentence>' in line:
                line = line.replace('</sentence>', '</sentence>\n\n')
            new_lines.append(line)
    with open(filename, 'w') as file:
        file.writelines(new_lines)
