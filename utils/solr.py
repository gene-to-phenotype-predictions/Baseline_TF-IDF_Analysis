import os as os
import sys as sys
import re as re
import json as json
import urllib as urllib
from requests import Request, Session

def select(core, query):
    
    url = f'http://localhost:8983/solr/{core}/select'
    
    s = Session()
    
    query_string = json.dumps(query) #urllib.parse.urlencode(query)
    
    request = Request('POST', url, data=query_string, headers={'Content-Type':'application/json'})
    
    request = s.prepare_request(request)
    
    response = s.send(request)
    
    try:
        data = json.loads(response.content)
    except:
        data = response.content

    return data