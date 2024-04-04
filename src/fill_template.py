from argparse import ArgumentParser
import re
from random import choice

TAGS = {
    '<pi>': 'pi',
    '<rp>': 'rp',
    '<shared entity>': 'se',
    '<type of entity>': 'te',
    '<relation>': 're'
}

def create_argparse() -> ArgumentParser:
    """
    Create an ArgumentParser object with the required arguments.

    Returns:
        argparse.ArgumentParser: The ArgumentParser object with the defined arguments.
    """
    aparse = ArgumentParser()
    aparse.add_argument('--template', '-T', type=str, required=True,
                        help='Path to the template file')
    aparse.add_argument('--relations', '-R', type=str, required=True,
                        help='Path to the relations file')
    aparse.add_argument('--output', '-O', type=str, default='output.txt',
                        help='Path to the output file')
    aparse.add_argument('--max-relations', '-L', type=int, default=100,
                        help='Maximum number of relations to consider')
    return aparse

def read_template(path) -> list:
    """
    Read the contents of a file and return them as a list of lines.

    Args:
        path (str): The path to the file to be read.

    Returns:
        list: A list of lines read from the file.
    """
    with open(path, 'r') as f:
        return f.readlines()

def read_relations(path) -> list:
    """
    Read the contents of a file located at the given path and return a list of lines.

    Args:
        path (str): The path to the file to be read.

    Returns:
        list: A list of strings, where each string represents a line from the file.
    """
    with open(path, 'r') as f:
        return f.readlines()

import re

def preprocess_template(template_rows) -> list:
    """
    Preprocesses a list of template rows by extracting tags and their positions.

    Args:
        template_rows (list): A list of strings representing the template rows.

    Returns:
        list: A list of tuples where each tuple contains the original row and a list of tags
              along with their start and end positions.

    Example:
        template_rows = [
            "This is a <pi> template.",
            "It contains <rp> tags for replacing values.",
            "The <shared entity> is used to represent a common entity.",
            "The <type of entity> specifies the type of the entity.",
            "The <relation> represents the relationship between entities."
        ]
        preprocess_template(template_rows)

        Output:
        [
            ("This is a <pi> template.", [("<pi>", 10, 14)]),
            ("It contains <rp> tags for replacing values.", [("<rp>", 12, 16)]),
            ("The <shared entity> is used to represent a common entity.", [("<shared entity>", 4, 19)]),
            ("The <type of entity> specifies the type of the entity.", [("<type of entity>", 4, 20)]),
            ("The <relation> represents the relationship between entities.", [("<relation>", 4, 14)])
        ]
    """
    pattern = re.compile(r'(<pi>|<rp>|<shared entity>|<type of entity>|<relation>){1,}')
    template_preprocessed = []
    for row in template_rows:
        row_tags = []
        for match in re.finditer(pattern, row):
            row_tags.append((match.group(), match.start(), match.end()))
        template_preprocessed.append((row, row_tags))
    return template_preprocessed

def replace_in_template(template_row, relation) -> str:
    """
    Replaces tags in a template with corresponding values from a relation.

    Args:
        template_row (tuple): A tuple containing two elements:
            - The template text as a list of characters.
            - A list of tags, where each tag is represented as a tuple with three elements:
                - The tag index in the template text.
                - The start index of the tag in the template text.
                - The end index of the tag in the template text.
        relation (dict): A dictionary containing the values to replace the tags with.
            The keys of the dictionary correspond to the tag indices.

    Returns:
        str: The template text with the tags replaced by their corresponding values from the relation.
    """
    global TAGS
    text = list(template_row[0])
    tags = template_row[1]
    for tag in tags:
        text[tag[1]] = f'<start_{TAGS[tag[0]]}> {relation[tag[0]]} <end_{TAGS[tag[0]]}>'
        text[tag[1]+1:tag[2]] = [''] * (tag[2] - tag[1] - 1)
        
    return ''.join(text)

def get_type_of_entity(entity):
    # bisogna guardare il pid su mapping di ml1m o lfm1m
    # in caso di U è user
    return 'USER' if entity[0] == 'U' else 'ENTITY' # TODO

def explode_relation(relation_text):
    """
    Explodes a relation text into a dictionary of template values.

    Args:
        relation_text (str): The relation text to be exploded.

    Returns:
        dict: A dictionary containing the exploded template values.
    """
    return {
        '<pi>': relation_text[2],
        '<rp>': relation_text[-1],
        '<shared entity>': relation_text[-3],
        '<type of entity>': get_type_of_entity(relation_text[-3]),
        '<relation>': relation_text[-2]
    }

if __name__ == '__main__':
    parser = create_argparse()
    args = parser.parse_args()
    
    template_rows = read_template(args.template)
    template_rows = preprocess_template(template_rows)

    relations = read_relations(args.relations)
    relations = [(explode_relation(expl := relation.strip().split(' ')), ' '.join(expl[:2]+[expl[-1]])) for relation in relations][:args.max_relations]

    finals = [(replace_in_template(choice(template_rows), rel[0]), rel[1]) for rel in relations]
    with open(args.output, 'w') as f:
        for line in finals:
            f.write(f'<start_rec> {line[1].strip()} <end_rec> <start_exp> {line[0].strip()[1:-1]} <end_exp>\n')
