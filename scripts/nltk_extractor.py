from typing import List, Any, Dict
import argparse

from tqdm import tqdm
import json
import os

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def extract_entities(
    text: str
) -> List[str]:
    """
    Extracts named entities from the given text.

    Args:
        text (str): The input text from which named entities will be extracted.

    Returns:
        list: A list of named entities extracted from the text.

    Example:
        >>> extract_entities("Apple Inc. is a technology company.")
        ['Apple Inc.']
    """
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Tag the words with their part-of-speech (POS) tags
    tagged_words = nltk.pos_tag(words)

    # Use the named entity chunker to extract named entities
    chunked_words = nltk.ne_chunk(tagged_words)

    # Extract the named entities from the chunked words
    entities = []
    for chunk in chunked_words:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))

    return entities


def argparser() -> argparse.Namespace:
    """Argument parser"""
    parser = argparse.ArgumentParser(description='Extract named entities from text.')
    parser.add_argument('--input_path', type=str, help='The file containing documents and summaries. The file must be a JSONL file (list of dictionaries) with keys "input_doc" for documents and "summary" for summaries.')
    parser.add_argument('--output_path', type=str, help='Where to save the signals. Signals are formatted as a JSONL file. Each dictionary has keys "doc_named_entities", "summary_named_entities", and "signal".')
    return parser.parse_args()


def load_jsonl_file(
    filepath: str
) -> List[Dict[str, Any]]:
    """Load a json into a list
    *arguments*
    *filepath* path to the file
    """
    data = []
    with open(filepath, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl_file(
    filepath: str,
    input_list: List[Any],
    mode: str = 'a+',
    overwrite: bool = False
) -> None:
    """Write a list into a jsonl
    *arguments*
    *filepath* path to save the file into
    *input_list* list to be saved in the json file, must be made of json iterable objects
    *overwrite* whether to force overwriting a file.
        When set to False you will append the new items to an existing jsonl file (if the file already exists).
    """
    if overwrite:
        try:
            os.remove(filepath)
        except:
            pass
    with open(filepath, mode, encoding='utf8') as writer:
        for line in input_list:
            writer.write(json.dumps(line) + '\n')


def main(args):
    data = load_jsonl_file(args.input_path)
    signals = []
    for line in tqdm(data):
        doc_ents = extract_entities(line['input_doc'])
        summary_ents = extract_entities(line['summary'])
        signal = list(set(doc_ents).intersection(set(summary_ents)))
        signals.append({
            'doc_named_entities': doc_ents,
            'summary_named_entities': summary_ents,
            'signal': signal
        })

    write_jsonl_file(args.output_path, signals, overwrite=True)


if __name__ == '__main__':
    args = argparser()
    main(args)
