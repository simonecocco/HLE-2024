from argparse import ArgumentParser
import re

TAGS = {
    '<pi>': 'pi',
    '<rp>': 'rp',
    '<shared entity>': 'se',
    '<type of entity>': 'te',
    '<relation>': 're'
}

def create_argparse():
    aparse = ArgumentParser()
    aparse.add_argument('--template', '-T', type=str, required=True,
                        help='Path to the template file')
    aparse.add_argument('--relations', '-R', type=str, required=True,
                        help='Path to the relations file')
    aparse.add_argument('--output', '-O', type=str,
                        help='Path to the output file')
    return aparse

def read_template(path):
    with open(path, 'r') as f:
        return f.readlines()

def preprocess_template(template_rows):
    pattern = re.compile(r'(<pi>|<rp>|<shared entity>|<type of entity>|<relation>){1,}')
    template_preprocessed = []
    for row in template_rows:
        row_tags = []
        for match in re.finditer(pattern, row):
            row_tags.append((match.group(), match.start(), match.end()))
        template_preprocessed.append((row, row_tags))
    return template_preprocessed

def replace_in_template(template_row, relation):
    pass # TODO

def get_type_of_entity(entity):
    pass # TODO

def explode_relation(relation_text):
    relation_text = relation_text.strip().split(' ')
    return {
        '<pi>': relation_text[2],
        '<rp>': relation_text[-1],
        '<shared entity>': relation_text[-3],
        '<type of entity>': 'ENTITY_TYPE',# TODO
        '<relation>': relation_text[-2]
    }

if __name__ == '__main__':
    parser = create_argparse()
    args = parser.parse_args()
    
    template_rows = read_template(args.template)
    template_rows = preprocess_template(template_rows)
