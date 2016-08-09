import requests

#URL = 'http://128.237.136.19:9001'
#URL = 'http://128.2.208.89:9001'
#URL = 'http://localhost:9001'
URL = 'http://tts.speech.cs.cmu.edu:9001'

#r = requests.get(URL)
#print r
#print r.headers
#print r.text

#payload = {'utt': 'tell me something'}
payload = {'utt': 'I like tom cruise'}
r = requests.post(URL, data=payload)
print r
print r.headers
print r.text
