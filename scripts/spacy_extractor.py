from typing import List, Any, Dict
import argparse

from tqdm import tqdm
import json
import os

import re
import spacy


def extract_entities(
    text: str
) -> List[str]:
    """
    Extracts named entities from the given text using Spacy.

    Args:
        text (str): The input text from which named entities will be extracted.

    Returns:
        list: A list of named entities extracted from the text.

    Example:
        >>> extract_entities("Apple Inc. is a technology company.")
        ['Apple Inc.']
    """
    # Load the Spacy English model
    nlp = spacy.load('en_core_web_sm')

    # Create a Spacy Doc object from the text
    doc = nlp(text)

    # Process the Doc with the Spacy model
    nlp.pipeline[0][1](doc)

    # Extract the named entities from the Doc
    if args.type == 'entity':
        entities = [ent.text for ent in doc.ents]
    elif args.type == 'noun_phrase':
        entities = [chunk.text for chunk in doc.noun_chunks]
    else:
        raise ValueError(f'Invalid entity type: {args.type}')

    return entities


def argparser() -> argparse.Namespace:
    """Argument parser"""
    parser = argparse.ArgumentParser(description='Extract named entities from text.')
    parser.add_argument('--input_path', type=str, help='The file containing documents and summaries. The file must be a JSONL file (list of dictionaries) with keys "input_doc" for documents and "summary" for summaries.')
    parser.add_argument('--output_path', type=str, help='Where to save the signals. Signals are formatted as a JSONL file. Each dictionary has keys "doc_named_entities", "summary_named_entities", and "signal".')
    parser.add_argument('--lower', action='store_true', help='Whether to make entities lower cased.')
    parser.add_argument('--type', default='entity', type=str, help='Possible values are "entity" and "noun_phrase".')
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


def process_entity(
    entity: str
) -> str:
    """Postprocess the entities."""
    pattern = r'\s*([A-Za-z])\.\s*([A-Za-z])'
    matches = re.finditer(pattern, entity)
    positions = [match.start() for match in matches]
    positions = [x+1 for x in positions]
    positions = [-1] + positions + [len(entity)]
    splitted_str = []
    for i in range(1, len(positions)):
        pos_start = positions[i-1] + 1
        pos_end = positions[i]
        splitted_str.append(entity[pos_start:pos_end])
    return '. '.join(splitted_str)


def main(args):
    data = load_jsonl_file(args.input_path)
    signals = []
    for line in tqdm(data):
        doc_ents = list(map(process_entity, extract_entities(line['input_doc'])))
        summary_ents = list(map(process_entity, extract_entities(line['summary'])))
        if args.lower:
            doc_ents = [x.lower() for x in doc_ents]
            summary_ents = [x.lower() for x in summary_ents]
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
