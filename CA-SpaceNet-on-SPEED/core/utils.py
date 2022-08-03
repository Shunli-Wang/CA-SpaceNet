import os
import io

import json
import trimesh
import random

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import copy

def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def load_bop_meshes(model_path):
    # load meshes
    meshFiles = [f for f in os.listdir(model_path) if f.endswith('.ply')]
    meshFiles.sort()
    meshes = []
    objID_2_clsID = {}
    for i in range(len(meshFiles)):  # i = 0
        mFile = meshFiles[i]
        objId = int(os.path.splitext(mFile)[0][4:])  # 1
        objID_2_clsID[str(objId)] = i  # {'1':0}
        meshes.append(trimesh.load(model_path + mFile))
        # print('mesh from "%s" is loaded' % (model_path + mFile))
    # 
    return meshes, objID_2_clsID

def load_bbox_3d(jsonFile):
    with open(jsonFile, 'r') as f:
        bbox_3d = json.load(f)
    bbox_3d = np.array(bbox_3d) * 5.86e-6
    return bbox_3d

def collect_mesh_bbox(meshpath, outjson, oriented=False):
    meshes, _ = load_bop_meshes(meshpath)
    allv = []
    for ms in meshes:
        if oriented:
            bbox = ms.bounding_box_oriented.vertices
        else:
            bbox = ms.bounding_box.vertices
        allv.append(bbox.tolist())
    with open(outjson, 'w') as outfile:
        json.dump(allv, outfile, indent=4)
        
def generate_shiftscalerotate_matrix(shift_limit, scale_limit, rotate_limit, width, height):
    dw = int(width * shift_limit)
    dh = int(height * shift_limit)
    pleft = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    shiftM = np.array([[1.0, 0.0, -pleft], [0.0, 1.0, -ptop], [0.0, 0.0, 1.0]])  # translation

    # random rotation and scaling
    cx = width / 2 # fix the rotation center to the image center
    cy = height / 2
    ang = random.uniform(-rotate_limit, rotate_limit)
    sfactor = random.uniform(-scale_limit, +scale_limit) + 1
    tmp = cv2.getRotationMatrix2D((cx, cy), ang, sfactor)  # rotation with scaling
    rsM = np.concatenate((tmp, [[0, 0, 1]]), axis=0)

    # combination
    M = np.matmul(rsM, shiftM)

    return M.astype(np.float32)

def draw_bounding_box(cvImg, R, T, bbox, intrinsics, color, thickness):
    rep = np.matmul(intrinsics, np.matmul(R, bbox.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    # bbox_lines = [0, 1, 0, 2, 0, 4, 5, 1, 5, 4, 6, 2, 6, 4, 3, 2, 3, 1, 7, 3, 7, 5, 7, 6]  # for swisscube
    bbox_lines = [0, 1, 0, 3, 0, 4, 1, 2, 1, 5, 2, 3, 2, 6, 3, 7, 4, 7, 4, 5, 5, 6, 6, 7]
    for i in range(int(len(bbox_lines)/2)):
        id1 = bbox_lines[2*i]
        id2 = bbox_lines[2*i+1]
        cvImg = cv2.line(cvImg, (x[id1],y[id1]), (x[id2],y[id2]), color, thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def draw_pose_axis(cvImg, R, T, bbox, intrinsics, thickness):
    radius = np.linalg.norm(bbox, axis=1).mean()
    aPts = np.array([[0,0,0],[0,0,radius],[0,radius,0],[radius,0,0]])
    rep = np.matmul(intrinsics, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0]/rep[2] + 0.5)
    y = np.int32(rep[1]/rep[2] + 0.5)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[1],y[1]), (0,0,255), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[2],y[2]), (0,255,0), thickness=thickness, lineType=cv2.LINE_AA)
    cvImg = cv2.line(cvImg, (x[0],y[0]), (x[3],y[3]), (255,0,0), thickness=thickness, lineType=cv2.LINE_AA)
    return cvImg

def get_single_bop_annotation(h_w, img_path, pose_json, pose_json_idx, objID_2_clsID):
    '''Get original info of the image'''

    # add attributes to function, for fast loading
    if not hasattr(get_single_bop_annotation, "dir_annots"):
        get_single_bop_annotation.dir_annots = {}

    # Load image
    # img_path ./data/SPEED/images/train/img002545.jpg
    height, width = h_w

    # Load image annotation
    idx = pose_json_idx.index(str(img_path.split('/')[-1]))
    img_ann = pose_json[idx]

    # Get the path of mask image
    img_mask_path = copy.deepcopy(img_path)
    img_mask_path.replace("images", "masks").replace("jpg", "png")
    gt_mask_visib = img_mask_path.replace("images", "masks").replace("jpg", "png")

    # Get K matrix
    objCnt = 1

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nu / 2],
         [0,     0,      1]]
    K = np.array(k)
    # K = np.array([3003.41297, 0.0, 960.0, 0.0, 3003.41297, 960.0, 0.0, 0.0, 1.0 ]).reshape(3,3)

    class_ids = []
    rotations = []
    translations = []
    merged_mask = np.zeros((height, width), np.uint8) # segmenation masks
    instance_idx = 1
    for i in range(objCnt):
        mask_vis_file = gt_mask_visib
        mask_vis = cv2.imread(mask_vis_file, cv2.IMREAD_UNCHANGED)  # (1200, 1920, 3)

        # padding
        padd_0 = np.zeros((360, 1920, 3)).astype(np.uint8)
        padd_1 = np.zeros((360, 1920, 3)).astype(np.uint8)
        # print(mask_vis.shape, padd.shape)
        mask_vis = np.concatenate((padd_0, mask_vis), axis=0)
        mask_vis = np.concatenate((mask_vis, padd_1), axis=0)
        
        # 
        R = np.transpose(quat2dcm(img_ann["q_vbs2tango"]))
        T = np.array(img_ann["r_Vo2To_vbs_true"]).reshape(3,1)
        obj_id = str(1)
        if not obj_id in objID_2_clsID:
            continue
        cls_id = objID_2_clsID[obj_id]  # 0
        # 
        class_ids.append(cls_id)
        rotations.append(R)
        translations.append(T)

        # compose segmentation labels
        merged_mask[mask_vis[:,:,0]==255] = instance_idx
        instance_idx += 1
    return K, merged_mask, class_ids, rotations, translations
    
def visualize_pred(img, gt, pred, mean, std):
    cvImg = img.to('cpu').numpy().transpose(1,2,0)
    # de-normalize
    cvImg = cvImg * (np.array(std).reshape(1,1,3) * 255)
    cvImg = cvImg + (np.array(mean).reshape(1,1,3) * 255)
    # 
    cvImg = cv2.cvtColor(cvImg.astype(np.uint8), cv2.COLOR_RGB2BGR)
    # cvImg[:] = 255

    cvRawImg = cvImg.copy()
    # 
    gtPoses = gt.to('cpu').to_numpy()

    gtVisual = gtPoses.visualize(cvImg)

    # show predicted poses
    for score, cls_id, R, T in pred:
        pt3d = np.array(gtPoses.keypoints_3d[cls_id])
        try:
            cvImg = draw_pose_axis(cvImg, R, T, pt3d, gtPoses.K, 2)
        except:
            pass

    return cvRawImg, cvImg, gtVisual

def remap_pose(srcK, srcR, srcT, pt3d, dstK, transM):
    '''
    @pt3d: [11, 3]
    '''
    ptCnt = len(pt3d)  # 11

    # print(srcK, dstK)
    ###
    # print('srcT', srcT)
    # pts_0 = np.matmul(srcK, np.matmul(srcR, pt3d.transpose()) + srcT)
    # xs_0 = pts_0[0] / (pts_0[2] + 1e-8)
    # ys_0 = pts_0[1] / (pts_0[2] + 1e-8)
    # print('xs_0:', xs_0)
    # print('ys_0:', ys_0)
    # print()
    ###

    pts = np.matmul(transM, np.matmul(srcK, np.matmul(srcR, pt3d.transpose()) + srcT))  # dstK * srcK ^ -1
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)
    # print('xs:', xs)
    # print('ys:', ys)
    # print()
    xy2d = np.concatenate((xs.reshape(-1,1), ys.reshape(-1,1)), axis=1)
    retval, rot, trans = cv2.solvePnP(pt3d.reshape(ptCnt,1,3), xy2d.reshape(ptCnt,1,2), dstK, None, flags=cv2.SOLVEPNP_EPNP)
    if retval:
        newR = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        newT = trans.reshape(-1, 1)

        newPts = np.matmul(dstK, np.matmul(newR, pt3d.transpose()) + newT)
        newXs = newPts[0] / (newPts[2] + 1e-8)
        newYs = newPts[1] / (newPts[2] + 1e-8)
        newXy2d = np.concatenate((newXs.reshape(-1,1),newYs.reshape(-1,1)), axis=1)
        diff_in_pix = np.linalg.norm(xy2d - newXy2d, axis=1).mean()

        return newR, newT, diff_in_pix
    else:
        print('Error in pose remapping!')
        return srcR, srcT, -1

# define a function which returns an image as numpy array from figure
def get_img_from_matplotlib_fig(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img

def visualize_accuracy_per_depth(
        accuracy_adi_per_class, 
        accuracy_rep_per_class, 
        accuracy_adi_per_depth, 
        accuracy_rep_per_depth, 
        depth_range):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    rep_keys = accuracy_rep_per_class[0].keys()
    adi_keys = accuracy_adi_per_class[0].keys()
    depth_bins = len(accuracy_rep_per_depth)
    assert(len(accuracy_adi_per_depth) == len(accuracy_rep_per_depth))
    ax1.set_title('Statistics of 2D error')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Success Rate (%)')
    ax2.set_title('Statistics of 3D error')
    ax2.set_xlabel('Depth')
    # ax2.set_ylabel('Success Rate (%)')
    # ax2.yaxis.tick_right()
    for k in rep_keys:
        xs = np.arange(depth_range[0], depth_range[1], (depth_range[1]-depth_range[0])/depth_bins)
        ys = []
        for i in range(depth_bins):
            if k in accuracy_rep_per_depth[i]:
                ys.append(accuracy_rep_per_depth[i][k])
            else:
                ys.append(0)
        ys = np.array(ys)
        # 
        # xnew = np.linspace(depth_range[0], depth_range[1], 300) / 1000
        # ynew = UnivariateSpline(xs, ys, k=2, s=100)(xnew)
        # ax1.plot(xnew, ynew, label=k)
        ax1.plot(xs, ys, marker='o', label=k)
    for k in adi_keys:
        xs = np.arange(depth_range[0], depth_range[1], (depth_range[1]-depth_range[0])/depth_bins)
        ys = []
        for i in range(depth_bins):
            if k in accuracy_adi_per_depth[i]:
                ys.append(accuracy_adi_per_depth[i][k])
            else:
                ys.append(0)
        ys = np.array(ys)
        # 
        # xnew = np.linspace(depth_range[0], depth_range[1], 300) / 1000
        # ynew = UnivariateSpline(xs, ys, k=2, s=100)(xnew)
        # ax2.plot(xnew, ynew, label=k)
        ax2.plot(xs, ys, marker='o', label=k)
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    ax1.grid()
    ax2.grid()
    matFig = get_img_from_matplotlib_fig(fig)
    # cv2.imshow("xx", matFig)
    # cv2.waitKey(0)
    return matFig
    
def print_accuracy_per_class(accuracy_adi_per_class,  accuracy_rep_per_class):
    assert(len(accuracy_adi_per_class) == len(accuracy_rep_per_class))
    classNum = len(accuracy_adi_per_class)

    firstMeet = True

    for clsIdx in range(classNum):
        if len(accuracy_adi_per_class[clsIdx]) == 0:
            continue

        if firstMeet:
            adi_keys = accuracy_adi_per_class[clsIdx].keys()
            rep_keys = accuracy_rep_per_class[clsIdx].keys()

            titleLine = "\t"
            for k in adi_keys:
                titleLine += (k + ' ')
            titleLine += '\t'
            for k in rep_keys:
                titleLine += (k + ' ')
            print(titleLine)

            firstMeet = False

        line_per_class = ("cls_%02d" % clsIdx)
        for k in adi_keys:
            line_per_class += ('\t%.2f' % accuracy_adi_per_class[clsIdx][k])
        line_per_class += '\t'
        for k in rep_keys:
            line_per_class += ('\t%.2f' % accuracy_rep_per_class[clsIdx][k])
        print(line_per_class)

def compute_pose_diff(mesh3ds, K, gtR, gtT, predR, predT):
    ptCnt = len(mesh3ds)

    pred_3d1 = (np.matmul(gtR, mesh3ds.T) + gtT).T
    pred_3d2 = (np.matmul(predR, mesh3ds.T) + predT).T

    p = np.matmul(K, pred_3d1.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d1 = p[:2].T

    p = np.matmul(K, pred_3d2.T)
    p[0] = p[0] / (p[2] + 1e-8)
    p[1] = p[1] / (p[2] + 1e-8)
    pred_2d2 = p[:2].T

    error_3d = np.linalg.norm(pred_3d1 - pred_3d2, axis=1).mean()
    error_2d = np.linalg.norm(pred_2d1 - pred_2d2, axis=1).mean()

    return error_3d, error_2d
