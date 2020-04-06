def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    listOfWords = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    listOfWords2=[];
    for word in listOfWords:
        listOfWords2.append(porter.stem(word))
    return ' '.join(listOfWords2)

allTweets_processed = text_process(allTweets)