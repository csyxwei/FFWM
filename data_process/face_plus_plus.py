# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import time
import json
from os.path import basename

http_url = 'https://api-cn.faceplusplus.com/facepp/v1/face/thousandlandmark'


def process(image_path, key, secret):
    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(image_path, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('all')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)
    for j, d in enumerate(data):
        if isinstance(d, str):
            data[j] = d.encode('utf-8')
    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        resp = urllib.request.urlopen(req, timeout=20)
        qrcont = resp.read()
        r = json.loads(qrcont.decode('utf-8'))
        if len(r['face']) == 0:
            print('Error', basename(image_path), 'no face !')
            return None
        return r
    except (urllib.error.HTTPError, Exception) as e:
        print('Error', basename(image_path), e)
        return None
