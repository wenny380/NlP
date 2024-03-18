import random


def read_in_files(file):
    name = open(file, 'r', encoding='utf-8').read().splitlines()
    return name


def splitdata(list_name):
    random.seed(42)
    random.shuffle(list_name)
    n1 = int(0.8 * len(list_name))
    X_tr = list_name[:n1]
    X_te = list_name[n1:]
    return X_tr, X_te


def write_to_file(filename, my_list):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in my_list:
            file.write(item + '\n')


words = read_in_files('female_names_rus.txt')
words1 = read_in_files('male_names_rus.txt')
words.extend(words1)
words = [word.strip() for word in words]
print(len(words))

Xtr, Xte = splitdata(words)

write_to_file("train.txt", Xtr)
write_to_file("test.txt", Xte)
