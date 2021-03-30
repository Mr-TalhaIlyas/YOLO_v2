import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
from random import randint, seed
from tensorflow.python.keras.utils.data_utils import Sequence
from data_processors import xyminmax_2_xywh
mpl.rcParams['figure.dpi'] = 300
# read_from_directory_train = '/home/user01/data_ssd/Talha/yolo_data/synth_fruit/train/'
# classes_name = []
# with open(read_from_directory_train + '_classes.txt', 'r') as f:
#         classes_name = classes_name + f.readlines()
def getAnchorIdx(label, anchors):
        """
        get the id of the best matching anchor for each ground truth box
        parameters:
            labels: tensor, of shape (num coords, 4) -> [x, y, w, h]
            anchors: tensor, of shape (num anchor, 2)
        returns:
            bestIdx: tensor, (conv H, convH)
        """
        anchors = anchors.astype(label.dtype)
        boxWH = label[..., 2:4]
        boxWH = np.tile(boxWH, anchors.shape[0])
        # duplicate the boxes so that we can calculate
        # its intersection with each anchor box
        boxWH = np.reshape(boxWH, (-1, anchors.shape[0], 2))
        # boxWH of shape (num box, num anchor, 2)
        overlap = np.minimum(boxWH, anchors)
        overlap = overlap[..., 0] * overlap[..., 1]
        # overlap of shape (num box, num anchor)
        boxArea = boxWH[..., 0] * boxWH[..., 1]
        anchorArea = anchors[..., 0] * anchors[..., 1]
        iou = overlap / (boxArea + anchorArea - overlap)
        # find the highest IOU and the anchor index
        bestIdx = np.argmax(iou, axis=-1)
        return bestIdx
    
def make_data_list(read_from_directory):
    
    train_datasets = []
    X_train = []
    Y_train = []
    
    with open(read_from_directory + '_annotations.txt', 'r') as f:
        train_datasets = train_datasets + f.readlines()
    
    for item in train_datasets:
      item = item.replace("\n", "").split(" ")
      tt = read_from_directory + item[0]
      X_train.append(tt)
      arr = []
      for i in range(1, len(item)):
        arr.append(item[i])
      Y_train.append(arr)
  
    return X_train, Y_train

def get_anchors(read_from_directory):
    file = open(read_from_directory + '_anchors.txt', 'r')
    anchors = file.readline().strip('\n')
    anchors = anchors.split(' ')
    y=[]
    for i in range(5):
        y.append([float(j) for j in anchors[i].split(',')])
    anchors = np.array(y)
    return anchors

def make_yolo_tensor(img_name, line, modelip_img_w, modelip_img_h, model_op_downsamples_by, anchors, num_class, true_box_buffer):
    
    
    grid_h = int(modelip_img_h / model_op_downsamples_by) # grid_size i.e. 13
    grid_w = int(modelip_img_w / model_op_downsamples_by)

    img = cv2.imread(img_name)
    orig_img_h, orig_img_w, _ = img.shape
    img = cv2.resize(img, (modelip_img_w, modelip_img_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    b_c_all = line # B_box co-ordinates
    label_matrix = np.zeros([grid_h, grid_w, num_class+5])
    
    
    all_coords = []
    for j in range(len(b_c_all)):
        b_c = np.asarray(b_c_all[j].split(','), np.uint)
        x_min, y_min, x_max, y_max, class_index = b_c
        # if the resolution in which you labelles image is different from model_ip
        # then we need to scale th b_boxes to new resolution (i.e. model_ip)
        if modelip_img_w != orig_img_w or modelip_img_h != orig_img_h:
            x_min, x_max = [i * (modelip_img_w / orig_img_w) for i in (x_min, x_max)]
            y_min, y_max = [i * (modelip_img_h / orig_img_h) for i in (y_min, y_max)]
        # This gives center_point(x,y) and (w,h) normalized corresponding to image hight and width 
        x, y, w, h = xyminmax_2_xywh(modelip_img_w, modelip_img_h, box = np.array([x_min, y_min, x_max, y_max]))
        all_coords.append([x, y, w, h, class_index])
        
    all_coords = np.array(all_coords).reshape(-1,5) # format [x,y,w,h,class_id]
    # scaling the boxes relative to op tensor size 
    scale = np.stack((grid_w, grid_h, grid_w, grid_h, 1), axis = -1)
    all_coords_scaled = all_coords * scale
    # instead on putting class index at last index we convert it to one_hot
    allcoords_onehotids = np.zeros((all_coords_scaled.shape[0], 4+num_class))
    allcoords_onehotids[..., 0:4] = all_coords_scaled[..., 0:4]
    for i, j in enumerate(all_coords_scaled[..., -1]):
        allcoords_onehotids[i, 4+int(j)] = 1
    # get the best anchor index for each bbox i.e. having max iou
    idx = getAnchorIdx(allcoords_onehotids, anchors)
    Y = np.floor(allcoords_onehotids[:, 1])  # extracting only integers
    X = np.floor(allcoords_onehotids[:, 0])  # extracting only integers
    numAnchor = anchors.shape[0]
    
    label_matrix = np.zeros((grid_h, grid_w, numAnchor, (num_class+5)))
    true_boxes = np.zeros((1, 1, 1, true_box_buffer, 4))
    for i, (y, x, anchor) in enumerate(zip(Y, X, idx)):
                    xInt, yInt, aInt = int(x), int(y), int(anchor)
                    filling_array = np.empty((1, num_class+5))
                    # box center coordinates
                    t_x = allcoords_onehotids[i][0] 
                    t_y = allcoords_onehotids[i][1]
                    # box width w.r.t to the anchor box
                    t_w = allcoords_onehotids[i][2] 
                    t_h = allcoords_onehotids[i][3] 
                    objscore_n_coords = np.array([t_x, t_y, t_w, t_h, 
                                                  1]) # indicate an object appears in this box
                    np.copyto(filling_array[:,0:5], objscore_n_coords)
                    np.copyto(filling_array[:,5:], allcoords_onehotids[i][4:])
                    label_matrix[yInt, xInt, aInt] = filling_array # we just filled this array with values
                    true_boxes[0, 0, 0, int(i%true_box_buffer)] = np.array([t_x, t_y, t_w, t_h])
                                                #  ^-- for handeling images which have more tha 50 b_boxes 
    return img, label_matrix, true_boxes

class My_Custom_Generator(Sequence) :
  
  def __init__(self, images, labels, batch_size, modelip_img_w, modelip_img_h,
               model_op_downsamples_by, anchors, num_class, true_box_buffer, shuffle=True):
    
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.modelip_img_w = modelip_img_w
    self.modelip_img_h = modelip_img_h
    self.model_op_downsamples_by = model_op_downsamples_by
    self.num_class = num_class
    self.anchors = anchors
    self.shuffle = shuffle
    self.true_box_buffer = true_box_buffer
    self.indices = np.arange(len(self.images))
    self.i = 0
    
  def on_epoch_end(self):
      # shuffling the indices
      if self.shuffle == True:
          np.random.shuffle(self.indices)
          # print('\n Shuffling Data...')
      
  def __len__(self) :
    # getting the total no. of iterations in one epoch
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    # from shuffled indices get the indices which will make the next batch 
    inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
    
    batch_x = []
    batch_y = []
    # loading data from those indices to arrays
    for i in inds:
        batch_x.append(self.images[i])
        batch_y.append(self.labels[i])
    
    train_image = []
    train_label = []
    train_true_boxes = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix, true_boxes = make_yolo_tensor(img_path, label, self.modelip_img_w, self.modelip_img_h,
                                             self.model_op_downsamples_by, self.anchors, self.num_class, self.true_box_buffer)
      train_image.append(image)
      train_label.append(label_matrix)
      train_true_boxes.append(true_boxes)
      # this  is just to save data in directory to see if generator is working
      # op = show_results((np.array(train_image)[0,::])*255, np.array(train_label)[0,::], classes_name, self.modelip_img_w, self.modelip_img_h)
      # cv2.imwrite('/home/user01/data_ssd/Talha/data_gen_test/img_{}.jpg'.format(self.i+1), op)
      # self.i = self.i+1
      
    return [np.array(train_image), np.array(train_true_boxes)], np.array(train_label)

#%%
# read_from_directory_train = 'C:/Users/Talha/Desktop/yolo_data/synth_fruit/train/'
# read_from_directory_val = 'C:/Users/Talha/Desktop/yolo_data/synth_fruit/valid/'
# grid_size = 7
# orig_img_w = 416 # original image resolution you used for labelling
# orig_img_h = 550
# modelip_img_w = 448
# modelip_img_h = 448
# num_class = 63
# batch_size = 4
# classes_name = []
# with open(read_from_directory_train + '_classes.txt', 'r') as f:
#         classes_name = classes_name + f.readlines()

# X_train, Y_train = make_data_list(read_from_directory_train)
# X_val, Y_val = make_data_list(read_from_directory_val)
# my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size, modelip_img_w, modelip_img_h,
#                                                   orig_img_w, orig_img_h, grid_size, num_class)
# my_validation_batch_generator = My_Custom_Generator(X_val, Y_val, batch_size, modelip_img_w, modelip_img_h,
#                                                   orig_img_w, orig_img_h, grid_size, num_class)

# x_train, y_train = my_training_batch_generator.__getitem__(0)
# x_val, y_val = my_training_batch_generator.__getitem__(0)
# print(x_train.shape)
# print(y_train.shape)

# print(x_val.shape)
# print(y_val.shape)

# #%%
# i = 3
# img = x_train[i, ...]  # batch * img_w * img_h * 3 => img_w * img_h * 3
# gt = y_train[i, ...] # batch * S * S * (num_class + 5) => S * S * (num_class+5)
# op = show_results(img, gt, classes_name, modelip_img_w, modelip_img_h)

# plt.imshow(op)


#%%
#@@@@@@@@@@@@@@@@
    # all_coords = np.array(all_coords).reshape(-1,5) # format [x,y,w,h,class_id]
    # # scaling the boxes relative to op tensor size 
    # scale = np.stack((grid_w, grid_h, grid_w, grid_h, 1), axis = -1)
    # all_coords_scaled = all_coords * scale
    # # instead on putting class index at last index we convert it to one_hot
    # allcoords_onehotids = np.zeros((all_coords_scaled.shape[0], 4+num_class))
    # allcoords_onehotids[..., 0:4] = all_coords_scaled[..., 0:4]
    # for i, j in enumerate(all_coords_scaled[..., -1]):
    #     allcoords_onehotids[i, 4+int(j)] = 1
    # # get the best anchor index for each bbox i.e. having max iou
    # idx = getAnchorIdx(allcoords_onehotids, anchors)
    # Y = np.floor(allcoords_onehotids[:, 1])  # extracting only integers
    # X = np.floor(allcoords_onehotids[:, 0])  # extracting only integers
    # numAnchor = anchors.shape[0]
    
    # label_matrix = np.zeros((grid_h, grid_w, numAnchor, (num_class+5)))
    # for i, (y, x, anchor) in enumerate(zip(Y, X, idx)):
    #                 xInt, yInt, aInt = int(x), int(y), int(anchor)
    #                 filling_array = np.empty((1, num_class+5))
    #                 objscore_n_coords = np.array([1,  # indicate an object appears in this box
    #                                                 # box center coordinates (tx, ty)
    #                                                 allcoords_onehotids[i][0] - x,
    #                                                 allcoords_onehotids[i][1] - y,
    #                                                 # box width w.r.t to the anchor box (tw, th)
    #                                                 allcoords_onehotids[i][2] / anchors[aInt][0],
    #                                                 allcoords_onehotids[i][3] / anchors[aInt][1]])
    #                 np.copyto(filling_array[:,0:5], objscore_n_coords)
    #                 np.copyto(filling_array[:,5:], allcoords_onehotids[i][4:])
    #                 label_matrix[yInt, xInt, aInt] = filling_array # we just filled this array with values
    # label_matrix = label_matrix.reshape(grid_h, grid_w, int(numAnchor*(num_class+5)))



# #@@
#     all_coords = np.array(all_coords).reshape(-1,5) # format [x,y,w,h,class_id]
#     # scaling the boxes relative to op tensor size 
#     scale = np.stack((grid_w, grid_h, grid_w, grid_h, 1), axis = -1)
#     all_coords_scaled = all_coords * scale
#     # get the best anchor index for each bbox i.e. having max iou
#     idx = getAnchorIdx(all_coords_scaled, anchors)
#     Y = np.floor(all_coords_scaled[:, 1])  # extracting only integers
#     X = np.floor(all_coords_scaled[:, 0])  # extracting only integers
#     numAnchor = anchors.shape[0]
    
#     label_matrix = np.zeros((grid_h, grid_w, numAnchor, 6))
#     for i, (y, x, anchor) in enumerate(zip(Y, X, idx)):
#         xInt, yInt, aInt = int(x), int(y), int(anchor)
#         label_matrix[yInt, xInt, aInt] = np.array([1,  # indicate an object appears in this box
#                                               # box center coordinates
#                                               all_coords_scaled[i][0] - x,
#                                               all_coords_scaled[i][1] - y,
#                                               # box width w.r.t to the anchor box
#                                               all_coords_scaled[i][2] / anchors[aInt][0],
#                                               all_coords_scaled[i][3] / anchors[aInt][1],
#                                               all_coords_scaled[i][4]])  # object class id










