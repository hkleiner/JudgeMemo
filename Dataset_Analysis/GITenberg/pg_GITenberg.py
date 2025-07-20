import json

from datasets import load_dataset


def check_id_in_ProjectGutenberg(idx):
    for book in pg_all:
        if str(idx) == str(book['id']):
            return True
    return False

pg_all = load_dataset("json", data_files="./ProjectGutenberg/cleaned_gutenberg.json", split="train")
path_pg = "../ProjectGutenberg/short_stories_8_16.json"
path_GITen = "books.txt"

GITenberg_books = open(path_GITen, 'r', encoding='utf-8')
ProjectGutenberg_stories = open(path_pg, 'r', encoding='utf-8')

pg_stories = json.load(ProjectGutenberg_stories)

# edit ids so that they match GITenberg IDs
for book in pg_all:
    book['id'] = book['id'].split('-')[0]

look_at_books = []
for book in GITenberg_books.readlines():
    idx = book.strip().split('_')[-1]  # title_pg-idx
    if check_id_in_ProjectGutenberg(book):
        print(f"{idx} in Project Gutenberg short stories")
    else:
        look_at_books.append(book)

with open("look_at_books_GITenberg.txt", 'w', encoding='utf-8') as file:
    for book in look_at_books:
        file.write(book)

GITenberg_books.close()
ProjectGutenberg_stories.close()
