
def my_split(sentence,separator):
    words = []
    word = ""
    for c in sentence:
        if c == separator:
            words.append(word)
            word = ""
        else:
            word = word+c
    words.append(word)
    return words
def my_join(words,separator):
    result = ""
    for i in range(len(words)):
        result = result + words[i]
        if i <len(words)-1:
            result = result + separator
    return result
sentence = input("Please enter sentence:")
parts = my_split(sentence," ")
print(my_join(parts,","))
for p in parts:
    print(p)
