import numpy as np
import torch 
from PIL import Image
import cv2

colorMap = np.array([[0, 0, 0],   # 0 "unlabeled", and others ignored 0
                    [100, 150, 245],    # 1 "car" 10   495
                    [100, 230, 245],      # 2 "bicycle" 11 575 [100, 230, 245]
                    [30, 60, 150],   # 3 "motorcycle" 15 棕色 240
                    [80, 30, 180],   # 4 "truck" 18 绛红 290
                    [100, 80, 250],    # 5 "other-vehicle" 20 红色 430
                    [255, 30, 30],   # 6 "person" 30 淡蓝色 315
                    [255,40,200],   # 7 "bicyclist" 31 淡紫色 [255,40,200]
                    [150, 30, 90],    # 8 "motorcyclist" 32 深紫色  270
                    [255, 0, 255],    # 9 "road" 40 浅紫色 510
                    [255, 150, 255],    # 10 "parking" 44 紫色 660
                    [75, 0, 75],   # 11 "sidewalk" 48 紫黑色
                    [175, 0, 75],   # 12 "other-ground" 49 深蓝色 250
                    [255, 200, 0],   # 13 "building" 50 浅蓝色 455
                    [255, 120, 50],   # 14 "fence" 51 蓝色 425
                    [0, 175, 0],   # 15 "vegetation" 70 绿色175
                    [135, 60, 0],   # 16 "trunk" 71 蓝色 195
                    [150, 240, 80],   # 17 "terrain" 72 青绿色 470
                    [255, 240, 150],   # 18 "pole"80 天空蓝 645
                    [255, 0, 0]   # 19 "traffic-sign" 81 标准蓝
                    ]).astype(np.int32) 

# colorMap = np.array([[0, 0, 0],         # 0 'noise' 1 
#                     [255, 120, 50],     # 1 'barrier'
#                     [100, 230, 245],    # 2  'bicycle'
#                     [135, 60, 0],       # 3  'bus'
#                     [100, 150, 245],    # 4  'car'
#                     [100, 80, 250],     # 5  'construction_vehicle'
#                     [30, 60, 150],      # 6  'motorcycle'
#                     [255, 30, 30],      # 7  'pedestrian'
#                     [255, 124, 128],    # 8   'traffic_cone'
#                     [255, 240, 150],    # 9   'trailer'
#                     [80, 30, 180],      # 10  'truck'
#                     [255, 0, 255],      # 11  'driveable_surface'
#                     [175, 0, 75],       # 12  'other_flat'
#                     [75, 0, 75] ,       # 13  'sidewalk'
#                     [150, 240, 80],     # 14  'terrain'
#                     [255, 200, 0],      # 15 'manmade'
#                     [0, 175, 0],        # 16  'vegetation'
            
#             ]).astype(np.int32)


def get_xy(size):
    """x 水平 y高低  z深度"""
    _x = np.zeros(size, dtype=np.int32)
    _y = np.zeros(size, dtype=np.int32)

    for i_h in range(size[0]):  # x, y, z
        _x[i_h, :] = i_h                 # x, left-right flip
    for i_w in range(size[1]):
        _y[:, i_w] = i_w                 # y, up-down flip
    
    return _x, _y


def point2cam_label(proj_label, image,img_filename):  #
    """_summary_

    Args:
        proj_label (_type_): torch.Tensor [1,370,1220]
        image (_type_): torch.Tensor [1,370,1220]
        mask_class (_type_): class number
        mask_color (_type_): list [100,150,245]
        img_filename (_type_): string
    """
    if type(proj_label) is torch.Tensor:
        
        proj_label = proj_label.cpu().numpy()  
        proj_label = proj_label.astype(np.int32) 
        
    if np.amax(proj_label) == 0:
        print('Oops! All voxel is labeled empty.')
        return
        
    # proj_label = proj_label.swapaxes(0,2)  
    # proj_label = proj_label.swapaxes(0,1)      
    # proj_label = proj_label.long()

    # get size
    size = proj_label.shape  
    # Convert to list
    proj_label = proj_label.flatten()
    # Get X Y Z
    _x, _y = get_xy(size)   
    _x = _x.flatten()
    _y = _y.flatten()
    
    # Get R G B
    proj_label[proj_label == 0] = 255  # empty 
    
    # Get X Y Z R G B
    xy_label = zip(_x, _y, proj_label[:])  # python2.7
    xy_label = list(xy_label)  # python3
    xy_label = np.array(xy_label)
    
    img_proj_label = np.zeros((size[0], size[1], 1), dtype=np.uint8)
    for i in range(len(proj_label)):
        img_proj_label[xy_label[i][0],xy_label[i][1],0] = xy_label[i][2]    
    
    # visualize
    img_proj_label[img_proj_label==255] = 0
    img_proj_label = img_proj_label.flatten()

    vis_label  = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(len(img_proj_label)):
        vis_label[xy_label[i][0],xy_label[i][1],0] = colorMap[img_proj_label[i]][2]        
        vis_label[xy_label[i][0],xy_label[i][1],1] = colorMap[img_proj_label[i]][1]
        vis_label[xy_label[i][0],xy_label[i][1],2] = colorMap[img_proj_label[i]][0]          
      
    
    
    # cv2.imwrite(img_filename,vis_label)
    kernel = np.ones((3,3),np.uint8)              #设置kenenel大小     
    # vis_label = cv2.dilate(vis_label,kernel,iterations=1) # 膨胀还原图形

    temp = np.sum(vis_label,axis=2)
    temp = temp[:,:,None]
    # proj_label = proj_label.long()
    ONE = np.ones_like(temp)
    ZERO = np.zeros_like(temp)
    proj_point_mask = np.where(temp!=0,ZERO,ONE)
    
    b_rgb = image
    b_rgb = image.swapaxes(0,2)
    b_rgb = b_rgb.swapaxes(0,1)
    print("b_rgb={}".format(b_rgb.shape))
    b_rgb[:,:,0] = (b_rgb[:,:,0]*0.229 + 0.485)
    b_rgb[:,:,1] = (b_rgb[:,:,1]*0.224+ 0.456)
    b_rgb[:,:,2] = (b_rgb[:,:,2]*0.225 + 0.406)
    b_rgb = b_rgb * 255
    
    b_rgb = b_rgb.cpu().numpy()
    
    # color 
    color_point_img = proj_point_mask * b_rgb + vis_label
    
    cv2.imwrite(img_filename,color_point_img) 
    
    # color_point_img = Image.fromarray(np.uint8(color_point_img))
    # color_point_img.save(img_filename)     
    
    
def point2cam(proj_point, image,img_filename):  #
    """_summary_

    Args:
        proj_label (_type_): torch.Tensor [1,370,1220]
        image (_type_): torch.Tensor [1,370,1220]
        mask_class (_type_): class number
        mask_color (_type_): list [100,150,245]
        img_filename (_type_): string
    """
    if type(proj_point) is torch.Tensor:
        
        proj_point = proj_point.cpu().numpy()  
       
        
    proj_point = np.sum(proj_point,axis=2)
    proj_point = proj_point[:,:,None]
    # proj_label = proj_label.long()
    ONE = np.ones_like(proj_point)
    ZERO = np.zeros_like(proj_point)

    proj_point_mask = np.where(proj_point!=0,ZERO,ONE)
    
    b_rgb = image.swapaxes(0,2)
    b_rgb = b_rgb.swapaxes(0,1)
    b_rgb[:,:,0] = (b_rgb[:,:,0]*0.229 + 0.485)
    b_rgb[:,:,1] = (b_rgb[:,:,1]*0.224+ 0.456)
    b_rgb[:,:,2] = (b_rgb[:,:,2]*0.225 + 0.406)
    b_rgb = b_rgb * 255
    
    b_rgb = b_rgb.cpu().numpy()
    
    # color 
    color_point_img = proj_point_mask * b_rgb + proj_point

    color_point_img = Image.fromarray(np.uint8(color_point_img))
    color_point_img.save(img_filename)     