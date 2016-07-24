### Preprocessing for NLU, including name entity recognition, customized NER
### Ting-Yao Hu, 2016.07
### benken.tyhu@gmail.com

import nltk

def tokenization(s):
    return nltk.word_tokenize(s)

def NER(tokens):
    tagged = nltk.pos_tag(tokens)
    namedEnt = nltk.ne_chunk(tagged)
    print namedEnt

def ngramStrExtract(tokens,n):
    l = len(tokens)-n+1
    ngrams = []
    for i in range(l):
        s = tokens[i]
        for j in range(n-1): s+=' '+tokens[i+j+1]
        ngrams.append(s)
    return ngrams

class MyNER(object):
    def __init__(self):
        self.types = []
        self.entityLists = {}

    def readNewType(self,fn):
        infile = file(fn,'r')
        typeName = infile.readline().strip()
        self.types.append(typeName)
        lst = []
        for line in infile: lst.append(line.lower().strip())
        self.entityLists[typeName] = lst


    ### detecting the entities appearing in s
    ### s -- input text string
    ### frame -- list of tuple ([type name],[value],[position],[length])
    def detection(self,s):
        tokens = tokenization(s)
        ngramlists = [ngramStrExtract(tokens,1),ngramStrExtract(tokens,2),ngramStrExtract(tokens,3)]
        frame = []
        for typeName in self.types:
            elist = self.entityLists[typeName]
            for entity in elist:
                for nidx, ngramlist in enumerate(ngramlists):
                    if entity in ngramlist:
                        frame.append((typeName,entity,ngramlist.index(entity),nidx+1))
                        break
        return frame


if __name__=='__main__':
    print 'test nlp_preprocessing'
    sp = '---------------------------------------------'

    ### tokenization
    print sp
    print 'Testing tokenization:'
    s = 'I like Tom Cruise'
    print 'input: '+s
    tokens = tokenization(s)
    print 'output: ', tokens
    
    """
    ### NER
    print sp
    print 'Testing NER'
    print 'input: '+s
    print 'NER:'
    NER(tokens)
    s2 = 'Who is steven spieberg?'
    print 'input: '+s2
    tokens = tokenization(s2)
    NER(tokens)
    """
    ### ngram string
    print sp
    print 'Testing ngram string extraction'
    s = 'I like Tom Cruise but not mat demon'
    tokens = tokenization(s)
    print ngramStrExtract(tokens,1)
    print ngramStrExtract(tokens,2)
    print ngramStrExtract(tokens,3)

    ### MyNER
    print sp
    print 'Testing MyNER'
    myner = MyNER()
    myner.readNewType('movie_example/actor.txt')
    myner.readNewType('movie_example/director.txt')
    myner.readNewType('movie_example/movie.txt')

    print myner.detection('I like tom cruise but not matt demon')   


