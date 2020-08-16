import json
import time
import _thread
import numpy as np
from os.path import basename, join
from face_plus_plus import process
import os


"""
process image to get landmarks
"""

def png2json(image, json_path=None, suf_num=4):
    base_name = basename(image)
    json_name = base_name[:-suf_num] + '.json'
    json_file = json_name
    if json_path is not None:
        json_file = join(json_path, json_name)
    return json_file


def get_key():
    key = ['k1', 'k2', 'k3']
    skey = ['s1', 's2', 's3']
    return key, skey


def multi_process(id, images, key, secret, suf_num=4):
    finished = 0
    total = len(images)
    print('-- {} -- starting process, totally {} files !'.format(id, total))
    for image in images:
        json_file = png2json(image, json_path, suf_num=suf_num)
        json_data = process(image, key, secret)
        if json_data is not None:
            with open(json_file, 'w') as f:
                json.dump(json_data, f)
            finished = finished + 1
        time.sleep(1)
        if finished % 100 == 0 and finished > 0:
            print('-- {} -- [{}/{}] finish '.format(id, finished, total))


if __name__ == '__main__':
    keys, skeys = get_key()

    # TODO change the path to your local path
    img_path = 'path/to/image'
    json_path = 'path/to/save json'
    suf_num = 7  # 7 for multipie and 4 for lfw and other dataset, 7 means to remove _xx.png and 4 means to remove .png/.jpg
    while True:
        images = os.listdir(img_path)
        images = [join(img_path, img) for img in images]
        # deduplicate
        images = [img for img in images if not os.path.exists(png2json(img, json_path, suf_num=suf_num))]

        if len(images) == 0:
            break
        print('Starting process, totally {} files !'.format(len(images)))

        key_num = len(keys)
        skip = int(len(images) / key_num)
        index = np.arange(0, key_num + 1) * skip
        index[-1] = len(images)
        print(index)

        try:
            for i in range(key_num - 1):
                _thread.start_new_thread(multi_process, (i, images[index[i]:index[i + 1]], keys[i], skeys[i], suf_num))
        except Exception as e:
            print('Error: ', e)
        multi_process(key_num - 1, images[index[-2]:], keys[0], skeys[0], suf_num)
