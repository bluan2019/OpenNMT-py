tag_dicts = {
        "< ent >": "<ent>",
        "< \ ent >": "<\ent>",
        "< N >": "<N>"
        }

def process_tag(text:str):
    for k, v in tag_dicts.items():
        text = text.replace(k, v)
    return text

if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    text = open(path, 'r').read()
    text = process_tag(text)
    print(text, file=open(path, 'w'), end="")
        

