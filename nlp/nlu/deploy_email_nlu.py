### Ting-Yao Hu, 2016.07
### benken.tyhu@gmail.com

import BaseHTTPServer
import time
import cgi
from nlu import *
from nlu_preprocessing import *



nlu = NLU()
nlu.readConfig('email_example/config')
nlu.initializeNER([])
nlu.train('email_example/corpus.txt')

#HOST_NAME = 'localhost'
HOST_NAME = ''
PORT_NUMBER = 9002
class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_POST(s):
        form = cgi.FieldStorage(
            fp = s.rfile,
            headers = s.headers,
            environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':s.headers['Content-Type']},
        )
        utt = form['utt'].value
        print 'input utterance: ',utt
        responseText = nlu.jsonStr(*nlu.understand(utt))
        print 'nlu result: ',responseText
        s.send_response(200)
        s.send_header('Content-type', 'text')
        s.end_headers()
        s.wfile.write(responseText)

if __name__=='__main__':
    server_class = BaseHTTPServer.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)
