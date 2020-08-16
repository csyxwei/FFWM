import os, cv2
import numpy as np
from os.path import basename, join
import json
from tqdm import tqdm

""" preprocess Multi-PIE and LFW dataset, save processed images. """

def s2f(file):
    """
    Get the corresponding frontal image name, only for Multi-PIE
    """
    path, name = os.path.split(file)
    ss = name.split('_')
    name = '{}_{}_{}_{}_{}'.format(ss[0], ss[1], ss[2], '051', ss[4])
    return name

def camera2ang(camera_id):
    """
    Get the angle corresponding to the camera, only for Multi-PIE
    """
    mapping = {'24_0': (+90, '10'), '01_0': (+75, '08'), '20_0': (+60, '08'), '19_0': (+45, '09'), '04_1': (+30, '07'),
               '05_0': (+15, '06'), '08_1': (-30, '05'), '19_1': (+45, '09'),
               # left
               '05_1': (0, '06'),  # center
               '14_0': (-15, '06'), '13_0': (-30, '05'), '08_0': (-45, '15'), '09_0': (-60, '15'), '12_0': (-75, '15'),
               '11_0': (-90, '15')}  # right
    return mapping[camera_id][0]

def resize_landmarks(landmarks, centerx, centery, r):
    """
    Resize the landmarks to load_size
    """
    landmarks[:, 0] = landmarks[:, 0] - (centerx - r)
    landmarks[:, 1] = landmarks[:, 1] - (centery - r)
    landmarks = landmarks * load_size / (r * 2)  # scale
    landmarks = landmarks.clip(min=0, max=load_size-1)
    return landmarks.astype('float32')

def get_valid_index(er, el, max_index):
    """
    Because the landmarks of occlusion part under large pose is wrong, we only use half landmarks under large poses.
    This method is used to get the index of valid part of the landmarks.
    :param er: the landmarks of right eye
    :param el: the landmarks of left eye
    :param max_index: the num of all landmarks
    :return: index of the valid landmarks
    """
    el_np, er_np = np.array(el), np.array(er)
    lev, rev = np.var(el_np[:, 0]), np.var(er_np[:, 0])
    r1, r2 = lev / (rev + 1e-10), rev / (lev + 1e-10)
    # If the ratio of the horizontal variance of the left and right eyes is less than 0.2,
    # we think that only half of the landmarks is valid
    if r1 > r2 and r2 <= 0.2:
        return (0, int(max_index / 2))
    elif r1 > r2 and r2 > 0.2:
        return (0, max_index)
    elif r1 < r2 and r1 <= 0.2:
        return (int(max_index / 2), max_index)
    else:
        return (0, max_index)

def json2np(json_file):
    """
    Read landmark json file, only for face++
    """
    f = open(json_file, 'r')
    landmarks = json.load(f)['face']['landmark']

    # face: right-up, right-low, left-up, left-low
    fru, frl, flu, fll = [], [], [], []
    for i in range(144):
        if i < 64:
            r = landmarks['face']['face_contour_right_{}'.format(i)]
            frl.append([r['x'], r['y']])
            l = landmarks['face']['face_contour_left_{}'.format(i)]
            fll.append([l['x'], l['y']])
        ld = landmarks['face']['face_hairline_{}'.format(i)]
        if i < 72:
            fru.append([ld['x'], ld['y']])
        else:
            flu.append([ld['x'], ld['y']])
    fr = frl + fru
    fl = flu + fll[::-1]

    # eyebrow: right, left
    ebr, ebl = [], []
    for i in range(64):
        l = landmarks['left_eyebrow']['left_eyebrow_{}'.format(i)]
        r = landmarks['right_eyebrow']['right_eyebrow_{}'.format(i)]
        ebl.append([l['x'], l['y']])
        ebr.append([r['x'], r['y']])

    # eye: right, left
    er, el = [], []
    for i in range(63):
        l = landmarks['left_eye']['left_eye_{}'.format(i)]
        r = landmarks['right_eye']['right_eye_{}'.format(i)]
        el.append([l['x'], l['y']])
        er.append([r['x'], r['y']])
    l = landmarks['left_eye']['left_eye_pupil_center']
    r = landmarks['right_eye']['right_eye_pupil_center']
    el.append([l['x'], l['y']])
    er.append([r['x'], r['y']])

    # eyelid: right, left
    edr, edl = [], []
    for i in range(64):
        l = landmarks['left_eye_eyelid']['left_eye_eyelid_{}'.format(i)]
        r = landmarks['right_eye_eyelid']['right_eye_eyelid_{}'.format(i)]
        edl.append([l['x'], l['y']])
        edr.append([r['x'], r['y']])

    # nose: right, left, middle
    nr, nl, nm = [], [], []
    for i in range(63):
        l = landmarks['nose']['nose_left_{}'.format(i)]
        r = landmarks['nose']['nose_right_{}'.format(i)]
        nl.append([l['x'], l['y']])
        nr.append([r['x'], r['y']])
    for i in range(60):
        ld = landmarks['nose']['nose_midline_{}'.format(i)]
        nm.append([ld['x'], ld['y']])
    l = landmarks['nose']['left_nostril']
    r = landmarks['nose']['right_nostril']
    nl.append([l['x'], l['y']])
    nr.append([r['x'], r['y']])

    # mouth: right, left
    mr, ml = [], []
    for i in range(64):
        u = landmarks['mouth']['upper_lip_{}'.format(i)]
        l = landmarks['mouth']['lower_lip_{}'.format(i)]
        if i < 16 or i >= 48:
            ml.extend([[u['x'], u['y']], [l['x'], l['y']]])
        else:
            mr.extend([[u['x'], u['y']], [l['x'], l['y']]])

    # crop center: the top point of the nose
    centerx = (landmarks['nose']['nose_left_0']['x'] +
               landmarks['nose']['nose_right_0']['x']) / 2
    centery = (landmarks['nose']['nose_left_0']['y'] +
               landmarks['nose']['nose_right_0']['y']) / 2

    leftx = landmarks['face']['face_contour_left_63']['x']
    rightx = landmarks['face']['face_contour_right_63']['x']

    return {'el': el, 'ml': ml, 'ebl': ebl, 'nl': nl, 'fl': fl,
            'er': er, 'mr': mr, 'ebr': ebr, 'nr': nr, 'fr': fr,
            'nm': nm, 'centerx': centerx, 'centery': centery,
            'leftx': leftx, 'rightx': rightx}

def get_extra_landmarks(face, key, cx, cy, r, max_l, mask_face):
    """
    Get extra hair and neck landmark of face.
    :param path: image path
    :param cx: the x-axis cord of face center
    :param cy: the y-axis cord of face center
    :param r: radius
    :param s: source image or target image, True is source
    :return: landmarks
    """
    idx = (0, max_l)
    fr, fl = np.array(face[0]), np.array(face[1])
    # get hair and neck landmarks, fxx -> face right/left up/low
    frl, fru = fr[:64], fr[64:]
    fll, flu = fl[64:], fl[:64]
    uy, uxr, uxl, ly, lxr, lxl = 1, 1, 1, 1, 1, 1
    s = key.split('_')[3]
    ang = camera2ang('{}_{}'.format(s[:2], s[2]))
    ratio = np.exp(np.cos(ang)) ** 2
    if ang > 0:
        uxr, lxr = uxr / ratio, lxr / ratio
        uxl, lxl = uxl * ratio, lxl * ratio
    elif ang < 0:
        uxr, lxr = uxr * ratio, lxr * ratio
        uxl, lxl = uxl / ratio, lxl / ratio
    pspace = 10  # Interval between two points
    landmarks = []
    for i in range(1, 15):
        if idx[0] == 0:
            # face-left-low -> left : hair and part neck
            landmarks.append(np.dstack([fll[0::pspace, 0] - i * lxl, fll[0::pspace, 1]])[0])
            # face-left-low -> down : neck
            landmarks.append(np.dstack([fll[0::pspace, 0], fll[0::pspace, 1] + i * ly])[0])
        if idx[1] == max_l:
            # face-right-low -> right : hair and part neck
            landmarks.append(np.dstack([frl[0::pspace, 0] + i * lxr, frl[0::pspace, 1]])[0])
            # face-right-low -> down : neck
            landmarks.append(np.dstack([frl[0::pspace, 0], frl[0::pspace, 1] + i * ly])[0])
    for i in range(1, 20):
        if idx[1] == max_l:
            # face-right-up -> up : hair
            landmarks.append(np.dstack([fru[0::pspace, 0], fru[0::pspace, 1] - i * uy])[0])
            # face-right-up -> right : hair
            landmarks.append(np.dstack([fru[0::pspace, 0] + i * uxr, fru[0::pspace, 1]])[0])
        if idx[0] == 0:
            # face-left-up -> up : hair
            landmarks.append(np.dstack([flu[0::pspace, 0], flu[0::pspace, 1] - i * uy])[0])
            # face-left-up -> left : hair
            landmarks.append(np.dstack([flu[0::pspace, 0] - i * uxl, flu[0::pspace, 1]])[0])
    landmarks = np.vstack(landmarks)
    landmarks = resize_landmarks(landmarks, cx, cy, r).astype('int')

    # remove the landmarks outside the face mask, implemented by a landmark mask (gate).
    mask = np.zeros((load_size, load_size))
    mask[landmarks[:, 1], landmarks[:, 0]] = 1
    mask = mask_face * mask
    gate = mask[landmarks[:, 1], landmarks[:, 0]] > 0
    return gate, landmarks

def merge(lm_face, lm_hair, idx_face, max_l, gate_hair):
    """
    Merge the face++ landmarks and extra hair and neck landmarks
    :param lm_face: face++ landmarks
    :param lm_hair: extra hair and neck landmarks
    :param idx_face: index gate for lm_face
    :param max_l: the total length of lm_face
    :param gate_hair: gate for lm_hair
    :return: landmarks
    """
    landmarks = np.vstack([lm_face, lm_hair])
    gate = []
    if idx_face[0] == 0:
        gate += [1] * (max_l // 2)
    else:
        gate += [0] * (max_l // 2)
    if idx_face[0] == max_l:
        gate += [1] * (max_l // 2)
    else:
        gate += [0] * (max_l // 2)
    gate += gate_hair.astype('float32').tolist()
    return landmarks, np.array(gate).astype('float32')

def image_transform(img_path, cx, cy, r, angle=0):
    """
    Crop and rotate the image or mask
    :param img_path: image path
    :param cx: the x-axis cord of face center
    :param cy: the y-axis cord of face center
    :param r: radius
    :param angle: is the image is frontal face image, rotate it by angle to make it horizontal
    :return: processed image or mask
    """

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    if angle != 0:
        mat = cv2.getRotationMatrix2D(center, int(angle / np.pi * 180), 1)
        img = cv2.warpAffine(img, mat, (w, h))
    # crop image
    img = img[max(0, cy - r):cy + r, max(0, cx - r): cx + r, :]
    if cx - r < 0:
        img = np.pad(img, ((0, 0), (r - cx, 0), (0, 0)), 'constant')
    if cx + r > w:
        img = np.pad(img, ((0, 0), (0, cx + r - w), (0, 0)), 'constant')
    if cy - r < 0:
        img = np.pad(img, ((r - cy, 0), (0, 0), (0, 0)), 'constant')
    if cy + r > h:
        img = np.pad(img, ((0, cy + r - h), (0, 0), (0, 0)), 'constant')
    img = cv2.resize(img, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    return img

def process_multipie_train(image_path, idx_face=None):
    """
    Process the multipie training dataset
    """
    landmarks_path = join(lm_dir, basename(image_path)[:-7] + '.json')
    mask_path = join(mask_dir, basename(image_path)[:-7] + '_07.png')
    key = basename(image_path)[:-7]
    lm_dict = json2np(landmarks_path)

    # calculate the rotation angle
    if key.split('_')[3] == '051':
        elc, erc = lm_dict['el'][-1], lm_dict['er'][-1]
        ang_tan = 1.0 * (elc[1] - erc[1]) / (elc[0] - erc[0])
        ang = np.arctan(ang_tan)
        keys = ['el', 'ml', 'ebl', 'nl', 'fl', 'er', 'mr', 'ebr', 'nr', 'fr', 'nm']
        for k in keys:
            v = np.array(lm_dict[k])
            x0 = v[:, 0] - 320
            y0 = v[:, 1] - 240
            # Note: the angle shoud be -ang to ensure it is consistent with opencv
            v[:, 0] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 320
            v[:, 1] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 240
            lm_dict[k] = v
        centerx, centery = lm_dict['centerx'], lm_dict['centery']  # the center cord of face
        x0 = centerx - 320
        y0 = centery - 240
        lm_dict['centerx'] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 320
        lm_dict['centery'] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 240
    else:
        ang = 0

    el, ml, ebl, nl, fl = lm_dict['el'], lm_dict['ml'], lm_dict['ebl'], lm_dict['nl'], lm_dict['fl']
    er, mr, ebr, nr, fr = lm_dict['er'], lm_dict['mr'], lm_dict['ebr'], lm_dict['nr'], lm_dict['fr']
    nm = lm_dict['nm']
    cx, cy = int(lm_dict['centerx']), int(lm_dict['centery'])  # crop center
    face = np.vstack([el, ml, ebl, nl, fl, nm, er, mr, ebr, nr, fr, nm]).astype('float32')

    maxy, miny = max(face[:, 1]), min(face[:, 1])
    r = int(max(abs(maxy - cy), abs(miny - cy))) # crop radius
    lm_face = resize_landmarks(face, cx, cy, r)

    mask = image_transform(mask_path, cx, cy, r, angle=ang)
    img = image_transform(image_path, cx, cy, r, angle=ang)
    mask = mask[:, :, 0]
    mask[mask > 0] = 255
    mask = mask.astype('uint8')

    max_l = face.shape[0]
    if idx_face is None: # for profile image
        idx_face = get_valid_index(er, el, face.shape[0])
        gate_hair, lm_hair = get_extra_landmarks((fr, fl), key, cx, cy, r, max_l, mask)
        lm_full, idx_full = merge(lm_face, lm_hair, idx_face, max_l, gate_hair)
        return lm_full, idx_full, idx_face, mask, img
    else: # for frontal image
        gate_hair, lm_hair = get_extra_landmarks((fr, fl), key, cx, cy, r, max_l, mask)
        lm_full, idx_full = merge(lm_face, lm_hair, idx_face, max_l, gate_hair)
        return lm_full, idx_full, idx_face, mask, img

def process_multipie_test(image_path):
    """
    Process the multipie testing dataset
    """
    landmarks_path = join(lm_dir, basename(image_path)[:-7] + '.json')
    key = basename(image_path)[:-7]
    lm_dict = json2np(landmarks_path)

    # calculate the rotation angle
    if key.split('_')[3] == '051':
        elc, erc = lm_dict['el'][-1], lm_dict['er'][-1]
        ang_tan = 1.0 * (elc[1] - erc[1]) / (elc[0] - erc[0])
        ang = np.arctan(ang_tan)
        keys = ['el', 'ml', 'ebl', 'nl', 'fl', 'er', 'mr', 'ebr', 'nr', 'fr', 'nm']
        for k in keys:
            v = np.array(lm_dict[k])
            x0 = v[:, 0] - 320
            y0 = v[:, 1] - 240
            # Note: the angle shoud be -ang to ensure it is consistent with opencv
            v[:, 0] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 320
            v[:, 1] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 240
            lm_dict[k] = v
        centerx, centery = lm_dict['centerx'], lm_dict['centery']  # the center cord of face
        x0 = centerx - 320
        y0 = centery - 240
        lm_dict['centerx'] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 320
        lm_dict['centery'] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 240
    else:
        ang = 0

    el, ml, ebl, nl, fl = lm_dict['el'], lm_dict['ml'], lm_dict['ebl'], lm_dict['nl'], lm_dict['fl']
    er, mr, ebr, nr, fr = lm_dict['er'], lm_dict['mr'], lm_dict['ebr'], lm_dict['nr'], lm_dict['fr']
    nm = lm_dict['nm']
    cx, cy = int(lm_dict['centerx']), int(lm_dict['centery'])  # crop center
    face = np.vstack([el, ml, ebl, nl, fl, nm, er, mr, ebr, nr, fr, nm]).astype('float32')
    maxy, miny = max(face[:, 1]), min(face[:, 1])
    r = int(max(abs(maxy - cy), abs(miny - cy))) # crop radius
    img = image_transform(image_path, cx, cy, r, angle=ang)

    return img


def process_lfw_test(image_path):
    """
    Process the lfw testing dataset
    """
    landmarks_path = join(lm_dir, basename(image_path)[:-4] + '.json')
    lm_dict = json2np(landmarks_path)

    # calculate the rotation angle
    elc, erc = lm_dict['el'][-1], lm_dict['er'][-1]
    ang_tan = 1.0 * (elc[1] - erc[1]) / (elc[0] - erc[0])
    ang = np.arctan(ang_tan)
    keys = ['el', 'ml', 'ebl', 'nl', 'fl', 'er', 'mr', 'ebr', 'nr', 'fr', 'nm']
    for k in keys:
        v = np.array(lm_dict[k])
        x0 = v[:, 0] - 125
        y0 = v[:, 1] - 125
        # Note: the angle shoud be -ang to ensure it is consistent with opencv
        v[:, 0] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 125
        v[:, 1] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 125
        lm_dict[k] = v
    centerx, centery = lm_dict['centerx'], lm_dict['centery']  # the center cord of face
    x0 = centerx - 125
    y0 = centery - 125
    lm_dict['centerx'] = x0 * np.cos(-ang) - y0 * np.sin(-ang) + 125
    lm_dict['centery'] = x0 * np.sin(-ang) + y0 * np.cos(-ang) + 125

    el, ml, ebl, nl, fl = lm_dict['el'], lm_dict['ml'], lm_dict['ebl'], lm_dict['nl'], lm_dict['fl']
    er, mr, ebr, nr, fr = lm_dict['er'], lm_dict['mr'], lm_dict['ebr'], lm_dict['nr'], lm_dict['fr']
    nm = lm_dict['nm']
    cx, cy = int(lm_dict['centerx']), int(lm_dict['centery'])  # crop center
    face = np.vstack([el, ml, ebl, nl, fl, nm, er, mr, ebr, nr, fr, nm]).astype('float32')
    maxy, miny = max(face[:, 1]), min(face[:, 1])
    r = int(max(abs(maxy - cy), abs(miny - cy)))  # crop radius
    img = image_transform(image_path, cx, cy, r, angle=ang)

    return img

if __name__ == '__main__':
    load_size = 128

    ############## Multi-PIE training set
    # img_dir = 'path/to/multipie train images'
    # mask_dir = 'path/to/multipie train masks'
    # lm_dir = 'path/to/multipie train landmark jsons'
    # save_dir = '../dataset/multipie/train/'
    # img_save_dir = join(save_dir, 'images')
    # mask_save_dir = join(save_dir, 'masks')
    # lm_save_path = join(save_dir, 'landmarks.npy')
    # if not os.path.exists(img_save_dir):
    #     os.makedirs(img_save_dir)
    # if not os.path.exists(mask_save_dir):
    #     os.makedirs(mask_save_dir)
    #
    # img_files = os.listdir(img_dir)
    # _landmarks = {'lm_S': {},
    #               'lm_F': {},
    #               'gate': {}}
    #
    # for img_file in tqdm(img_files):
    #     if img_file.split('_')[3] in ['081', '191']:
    #         continue
    #     landmarks_path = join(lm_dir, basename(img_file)[:-7] + '.json')
    #     if not os.path.exists(landmarks_path):
    #         continue
    #
    #     path_S, path_F = join(img_dir, img_file), join(img_dir, s2f(img_file))
    #     key_S = basename(path_S)[:-7]
    #     key_F = basename(path_F)[:-7]
    #     lm_S, gate_S, idx_face, mask_S, img_S = process_multipie_train(path_S)
    #     lm_F, gate_F, _, mask_F, img_F = process_multipie_train(path_F, idx_face)
    #     gate = gate_S * gate_F
    #     _landmarks['lm_S'][key_S] = lm_S
    #     _landmarks['lm_F'][key_F] = lm_F
    #     _landmarks['gate'][key_S] = gate
    #     cv2.imwrite(join(img_save_dir, basename(path_S)), img_S)
    #     cv2.imwrite(join(img_save_dir, basename(path_F)), img_F)
    #     cv2.imwrite(join(mask_save_dir, basename(path_S)), mask_S)
    #     cv2.imwrite(join(mask_save_dir, basename(path_F)), mask_F)
    # np.save(lm_save_path, _landmarks)


    ########### Multi-PIE testing set
    # img_dir = 'path/to/multipie test images'
    # lm_dir = 'path/to/multipie test landmark jsons'
    # save_dir = '../dataset/multipie/test/'
    # img_save_dir = join(save_dir, 'images')
    # if not os.path.exists(img_save_dir):
    #     os.makedirs(img_save_dir)
    # img_files = os.listdir(img_dir)
    # for img_file in tqdm(img_files):
    #     if img_file.split('_')[3] in ['081', '191']:
    #         continue
    #     landmarks_path = join(lm_dir, basename(img_file)[:-7] + '.json')
    #     if not os.path.exists(landmarks_path):
    #         continue
    #     path_S = join(img_dir, img_file)
    #     img_S = process_multipie_test(path_S)
    #     cv2.imwrite(join(img_save_dir, basename(path_S)), img_S)

    ########### LFW testing set
    # img_dir = 'path/to/lfw images'
    # lm_dir = 'path/to/lfw landmark jsons'
    # save_dir = '../dataset/lfw'
    # img_save_dir = join(save_dir, 'images')
    # if not os.path.exists(img_save_dir):
    #     os.makedirs(img_save_dir)
    # img_files = os.listdir(img_dir)
    # for img_file in tqdm(img_files):
    #     landmarks_path = join(lm_dir, basename(img_file)[:-4] + '.json')
    #     if not os.path.exists(landmarks_path):
    #         continue
    #     path_S = join(img_dir, img_file)
    #     img_S = process_lfw_test(path_S)
    #     cv2.imwrite(join(img_save_dir, basename(path_S)), img_S)
