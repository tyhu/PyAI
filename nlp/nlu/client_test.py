import requests

#URL = 'http://128.237.136.19:9001'
URL = 'http://localhost:9001'

#r = requests.get(URL)
#print r
#print r.headers
#print r.text

payload = {'utt': 'tell me something'}
r = requests.post(URL, data=payload)
print r
print r.headers
print r.text
