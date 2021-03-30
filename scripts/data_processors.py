import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randint, seed
import tensorflow as tf


#%%
def convert_labels(label_matrix, grid_h, grid_w, numAnchor, num_class):
    '''
    label_matrix : shape (grid_w, grid_h, numAnchors * (num_class+5))
    label_matrix_small : shape (grid_w, grid_h, numAnchors, 6)
    '''
    #  From 13x13x5x25  -->  13x13x5x6
    label_matrix_small = np.zeros((grid_h, grid_w, numAnchor, 6))
    for i in range(5):
        x= label_matrix[:,:,i:i+1,5:]
        y = np.argmax(x, axis= -1)
        y = y[:,:, np.newaxis, :]
        label_matrix_small[:,:,i:i+1,:] = np.concatenate((label_matrix[:,:,i:i+1,0:5], y), axis=-1)
        
    return label_matrix_small
#%%
def label2Box(labels, grid_size):
    
    # creating mask form objectness scores we put while making tensor
    mask = np.squeeze(labels[..., 4:5]).astype(np.bool)
    # getting class ids
    classId = labels[..., 5:6]
    # seperating x,y,w,h coordinates
    labels = np.squeeze(labels[..., 0:4])

    boxXY = labels[..., 0:2] / grid_size
    # Each of the anchor will scale the tw and th in
    # it's respective channel
    boxWH = labels[..., 2:4] / grid_size 

    labels = np.concatenate((classId, boxXY - boxWH / 2, boxXY + boxWH / 2), axis=-1)
    labels = labels[mask]
    
    return labels

#%%
def xywh_2_xyminmax(img_w, img_h, box):
    '''
    input_box : (x, y, w, h)
    output_box : (xmin, ymin, xmax, ymax) @ un_normalized
    '''
    xmin = box[0] - (box[2] / 2)
    ymin = box[1] - (box[3] / 2)
    xmax = box[0] + (box[2] / 2)
    ymax = box[1] + (box[3] / 2)
    
    box_minmax = np.array([xmin*img_w, ymin*img_h, xmax*img_w, ymax*img_h])
    
    return box_minmax
#%%
def xyminmax_2_xywh(img_w, img_h, box):
    '''
    input_box  : (xmin, ymin, xmax, ymax)
    output_box : (x, y, w, h) @ normalized
    '''
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = np.abs(box[2] - box[0])
    h = np.abs(box[3] - box[1])
    
    x = x * (1./img_w)
    w = w * (1./img_w)
    y = y * (1./img_h)
    h = h * (1./img_h)
    
    return (x,y,w,h)

#%%   
def IoU(target_boxes , pred_boxes):
    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / (( boxAArea + boxBArea - interArea ) + 1e-7) # avoid division by zero
    iou = np.nan_to_num(iou)
    return iou

#%%
def sigmoid(x):
    '''applies sigmoid activation element-wise'''
    x = 1/(1 + np.exp(-x)) 
    return x

def exp(x):
    '''takes exponent element-wise'''
    x = np.exp(x)
    return x

def softmax(x, axis=-1, t=-100.):
    '''applies softmax activation along given axis'''
    x = x - np.max(x)
    if np.min(x) < t:
        x = x/np.min(x)*t
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

def get_translation_matrix(pred, anchors):
    '''
    Parameters
    ----------
    pred : output tensor of yolo_v2; shape (grid_h, gird_w, num_anchors, (5+classes))
    Returns
    -------
    Matrices for shifitng coordinates from range [0,1] to [0, grid_size]
    for equation in yolo_v2 paper
        b_x = sigmoid(t_x) + c_x
        b_w = p_w * exp(t_w)
    mat_grid_h : is c_y; shape (13,13,5)
    mat_grid_w : is c_x; shape (13,13,5)
    mat_anchor_h : is p_h; shape (13,13,5)
    mat_anchor_w : is p_w; shape (13,13,5)
    '''
    grid_h, grid_w, num_anchors = pred.shape[:3]
    # seperate anochors width and hight
    anchors_w = anchors[::2]
    anchors_h = anchors[1::2]
    
    mat_grid_h = np.zeros((grid_h, grid_w, num_anchors))
    for i in range(grid_h):
        mat_grid_h[i,:,:] = i
    mat_grid_w = np.zeros((grid_h, grid_w, num_anchors))
    for i in range(grid_w):
        mat_grid_w[:,i,:] = i
        
    mat_anchor_h = np.zeros((grid_h, grid_w, num_anchors))
    for i in range(num_anchors):
        mat_anchor_h[:,:, i] = anchors_h[i]
    mat_anchor_w = np.zeros((grid_h, grid_w, num_anchors))
    for i in range(num_anchors):
        mat_anchor_w[:,:, i] = anchors_w[i]
        
    return mat_grid_h, mat_grid_w, mat_anchor_h, mat_anchor_w
#%
def adjust_pred(pred, anchors):
    '''
    pred : output tensor of yolo_v2; shape (grid_h, gird_w, num_anchors, (5+classes))

    This function applies appropriate activations on yolo_v2 op and translate and noramlize and box coord,
    and corrects the class_pr  (i.e. class_pr * obj_score)
    '''
    grid_h, grid_w, num_anchors = pred.shape[:3]
    # get translation matrices
    mat_grid_h, mat_grid_w, mat_anchor_h, mat_anchor_w = get_translation_matrix(pred, anchors)
    '''
    first convert range appropriately from [0,1] to  [0,13],
    then normalize between [0,1] again
        b_x = sigmoid(t_x) + c_x
        b_w = p_w * exp(t_w)
    '''
    pred[..., 0] = (sigmoid(pred[..., 0]) + mat_grid_w) / grid_w # b_x
    pred[..., 1] = (sigmoid(pred[..., 1]) + mat_grid_h) / grid_h # b_y
    pred[..., 2] = (exp(pred[..., 2]) * mat_anchor_w) / grid_w   # b_w
    pred[..., 3] = (exp(pred[..., 3]) * mat_anchor_h) / grid_h   # b_h
    
    # This is just a hack i used to normalize and rescale obj__score values
    pred[..., 4] = sigmoid(np.abs((pred[..., 4]/(np.max(np.abs(pred[..., 4])))))) # rescaling objectness/confidence scores
    #pred[..., 4] = sigmoid(pred[..., 4]) # rescaling objectness/confidence scores
    
    obj_score = pred[...,4]
    obj_score = obj_score[...,np.newaxis]# expand dims for matching dims wiht class_pr
    pred[..., 5:] =  softmax(pred[..., 5:])# * obj_score 
    
    return pred

def extract_boxes(pred_adj, confd_thresh, img_h, img_w):
    box_coord = pred_adj[..., 0:4]
    # i already multiplitd the class_pr and obj_score during adjust_pred, so following is now conf_score
    confd_scores = pred_adj[..., 5:] 
    
    # returns the indices of cell which highest confidence of having an object
    detected_classes = np.argmax(confd_scores, axis=-1)
    # returns the confidence score of a class in a cell
    class_scores = np.max(confd_scores, axis=-1)
    
    # zeroing out the less confident cells
    mask = (class_scores >= confd_thresh)
    # applying mask to the class_scores and 
    detected_classes = detected_classes[mask]
    class_scores = class_scores[mask]
    box_coord = box_coord[mask] # boxes are in xy_wh normalized format
    b_boxes_scaled = np.empty((box_coord.shape))
    for i in range(box_coord.shape[0]):
        b_boxes_scaled[i,:] = xywh_2_xyminmax(img_h, img_w, box_coord[i,:])
        
    return b_boxes_scaled, detected_classes, class_scores




def nms(b_boxes_scaled, detected_classes, class_scores, classes_name, iou_thresh=0.5, keep_boxes=7):
    # get indices of max class scores
    sorted_class_scores_ind = np.argsort(class_scores)[::-1]
    
    # now sort all the coords, classes and scores in descending score order
    class_scores = class_scores[sorted_class_scores_ind]
    b_boxes_scaled = b_boxes_scaled[sorted_class_scores_ind,:]
    detected_classes = detected_classes[sorted_class_scores_ind]
    
    for i in range(b_boxes_scaled.shape[0]):
        for j in range(i+1, b_boxes_scaled.shape[0], 1):
            box_max = b_boxes_scaled[i]
            box_cur = b_boxes_scaled[j]
            iou = IoU(box_max, box_cur)
            #print(iou)
            if iou > iou_thresh:
                b_boxes_scaled[j,:] = 0
                class_scores[j] = 0
                detected_classes[j] = 0
    # get non zero indices
    non_zero_ind = np.nonzero(class_scores)
    
    nms_scores = class_scores[non_zero_ind]
    nms_boxes = b_boxes_scaled[non_zero_ind]
    nms_classes = detected_classes[non_zero_ind]
    nms_classes_names = np.take(classes_name, nms_classes)
    
    # only keep top K boxes defined by 'keep_boxes' = 10
    nms_boxes = nms_boxes[0:keep_boxes,:]
    nms_scores = nms_scores[0:keep_boxes]
    nms_classes_names = nms_classes_names[0:keep_boxes]
    
    return nms_scores, nms_boxes, nms_classes_names

def get_boxes(pred_adj, confd_thresh, img_h, img_w):
    '''
    Returns:
    -------
    b_boxes:          array of (-1,4), containing [x, y, w, h]
    detected_classes: array of (-1,1), containing int [class_indices]
    class_scores:     array of (-1,1), containing float [class_scores] 
    '''
    grid_h, grid_w, num_anchors = pred_adj.shape[:3]
    b_boxes = []
    detected_classes = []
    class_scores = []
    
    for g_h in range(grid_h):
        for g_w in range(grid_w):
            for a in range(num_anchors):
                class_pr = pred_adj[g_h,g_w,a,5:]   # class probabilities
                confd = pred_adj[g_h,g_w,a,4]       # confidence/objectness score
                x, y, w, h = pred_adj[g_h,g_w,a,:4] # b_box coords
                detected_class = np.argmax(class_pr)
                
                if class_pr[detected_class] > confd_thresh:
                    b_boxes.append((x, y, w, h))
                    detected_classes.append(detected_class)
                    class_scores.append(class_pr[detected_class])
    b_boxes = np.array(b_boxes) 
    # scale boses w.r.t network input output     
    b_boxes_scaled = np.zeros((b_boxes.shape))
    for i in range(b_boxes.shape[0]):
        b_boxes_scaled[i,:] = xywh_2_xyminmax(img_h, img_w, b_boxes[i,:])
        
    return np.array(b_boxes_scaled), np.array(detected_classes), np.array(class_scores) 