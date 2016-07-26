### Natrual language understanding
### Ting-Yao Hu, 2016.07
### benken.tyhu@gmail.com

import json
from nlu_preprocessing import *

class NLU(object):
    def __init__(self):
        self.intentList = []
        self.typeDict = {}

    def readConfig(self,fn):
        cfgfile = file(fn,'r')
        self.cfg = json.load(cfgfile)
        #print self.cfg

    def train(self,corpus_fn):
        self.labs = [obj['label'] for obj in self.cfg]
        self.slotTypeDicts = {}
        for obj in self.cfg:
            if 'entities' in obj.keys():
                self.slotTypeDicts[obj['label']] = [entity['type'] for entity in obj['entities']]
            else: self.slotTypeDicts[obj['label']] = []
        corpus = json.load(file(corpus_fn,'r'))
        intentList,featList = [],[]
        for ex in corpus:
            intentList.append(ex['intent'])
            s = ex['utt']
            tokens = tokenization(s)
            feat = self.lexicalFeatExtract(tokens)
            feat+=self.lexicalFeatExtract(self.transformUtt(s))
            featList.append(feat)
        self.featList, self.intentList = featList, intentList

    def initializeNER(self,fnlist):
        self.ner = MyNER()
        for fn in fnlist: self.ner.readNewType(fn)

    ### transformUtt:
    ### replace entity to entity tag
    def transformUtt(self, s):
        tokens = tokenization(s)
        slots = self.ner.detection(s)
        for slot in slots:
            typename,_,pos,l = slot
            for i in range(pos,pos+l):
                tokens[i] = '['+typename+']'
        return tokens

    def lexicalFeatExtract(self,tokens):
        ngramfeat = []
        for n in range(1,4):
            ngramfeat+=ngramStrExtract(tokens,n)
        return ngramfeat

    def getMatchNum(self,feat1,feat2):
        i = 0
        for f in feat1:
            for f2 in feat2:
                if f==f2:
                    i+=1
                    break
        return i
        
    def understand(self,s):
        slots = self.ner.detection(s)
        tokens = tokenization(s)
        test_feat = self.lexicalFeatExtract(tokens)
        test_feat+=self.lexicalFeatExtract(self.transformUtt(s))
        maxscore = 0
        maxidx = 0
        for idx,feat,intent in zip(range(len(self.featList)),self.featList,self.intentList):
            score = float(self.getMatchNum(test_feat,feat))/len(feat)
            if maxscore<score:
                maxidx = idx
                maxscore = score

        test_intent = self.intentList[maxidx]
        lst = self.slotTypeDicts[test_intent]
        test_slots = []
        for slot in slots:
            if slot[0] in lst: test_slots.append(slot)
        return  test_intent, test_slots

    def jsonStr(self,intent,slots):
        obj = {}
        obj['label'] = intent
        obj['entities'] = slots
        return json.dumps(obj)
        
        
        

if __name__=='__main__':
    sp = '------------------------------------------'
    
    print sp
    print 'Testing NLU:'
    nlu = NLU()
    nlu.readConfig('movie_example/config')
    print nlu.cfg
    print 'initialize NER:'
    nlu.initializeNER(['movie_example/actor.txt','movie_example/director.txt','movie_example/movie.txt'])
    print 'transformUtt:'
    s = 'I like tom cruise but not matt demon'
    print nlu.transformUtt(s)
    print nlu.lexicalFeatExtract(tokenization(s))
    s2 = ' '.join(nlu.transformUtt(s))
    print nlu.lexicalFeatExtract(s2.split())
    print 'model training'
    nlu.train('movie_example/corpus.txt')
    print 'understanding:'
    nlu.understand(s)

    print 'json string:'
    print nlu.jsonStr(*nlu.understand(s))
