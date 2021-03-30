import tensorflow as tf
import numpy as np
if int(str(tf.__version__)[0]) == 2:
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.applications import Xception, MobileNetV2
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D, Dense, LeakyReLU
    from tensorflow.keras.layers import MaxPooling2D, concatenate, Dropout, Flatten, Reshape, Concatenate, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    from tensorflow.keras.backend import resize_images, int_shape
    import tensorflow.keras.backend as K
if int(str(tf.__version__)[0]) == 1:
    from keras.models import Model
    from keras.regularizers import l2, l1
    from keras.applications.mobilenetv2 import MobileNetV2
    from keras.applications.xception import Xception
    from keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dropout, Flatten, Dense, Reshape
    from keras.layers import MaxPooling2D, concatenate, Concatenate, LeakyReLU, SeparableConv2D, GlobalAveragePooling2D
    from keras.backend import resize_images, int_shape
    from keras.optimizers import Adam, Nadam, SGD
    import keras.backend as K

#%%
def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)

def Darknet_19(input_img, true_boxes, boxes_percell, num_class, dropout, as_classifier = False, useBias = True, Alpha = 0.1):
        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_img)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        skip_connection = x
        
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)
        
        x = concatenate([skip_connection, x])
        
        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 23
        x = Conv2D(5 * (4 + 1 + 20), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
        output = Reshape((13, 13, 5, 4 + 1 + 20))(x)
        
        # small hack to allow true_boxes to be registered when Keras build the model 
        # for more information: https://github.com/fchollet/keras/issues/2790
        output = Lambda(lambda args: args[0])([output, true_boxes])
        model = Model([input_img, true_boxes], output)
        
        return model
    


#%%
def Darknet_19_mine(input_img, true_boxes, boxes_percell, num_class, dropout, as_classifier = False, useBias = False, Alpha = 0.1):
    # Entry FLow
    #input
    ip = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal',  use_bias=False, padding = 'same', name = 'ip_conv')(input_img)
    ip = BatchNormalization(name= 'ip_conv_bn')(ip)
    ip = LeakyReLU(alpha=Alpha, name= 'ip_conv_act')(ip)
    ip = MaxPooling2D(name= 'ip_conv_pool')(ip)
    
    c2 = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', use_bias=False, padding = 'same', name = 'conv_2')(ip)
    c2 = BatchNormalization(name = 'conv_2_bn')(c2)
    c2 = LeakyReLU(alpha=Alpha, name = 'conv_2_act')(c2)       # *******___1/4 times smaller than ip___********
    c2 = MaxPooling2D(name = 'conv_2_pool')(c2)
    # Block 1
    c3 = Conv2D(filters = 128, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_31')(c2)
    c3 = BatchNormalization(name = 'conv_31_bn')(c3)
    c3 = LeakyReLU(alpha=Alpha, name = 'conv_31_act')(c3)
    c3 = Conv2D(filters = 64, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_32')(c3)
    c3 = BatchNormalization(name = 'conv_32_bn')(c3)
    c3 = LeakyReLU(alpha=Alpha, name = 'conv_32_act')(c3)
    c3 = Conv2D(filters = 128, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_33')(c3)
    c3 = BatchNormalization(name = 'conv_33_bn')(c3)
    c3 = LeakyReLU(alpha=Alpha, name = 'conv_33_act')(c3)       # *******___1/4 times smaller than ip___********
    c3 = MaxPooling2D(name = 'conv_3_pool')(c3)
    
    c4 = Conv2D(filters = 256, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_41')(c3)
    c4 = BatchNormalization(name = 'conv_41_bn')(c4)
    c4 = LeakyReLU(alpha=Alpha, name = 'conv_41_act')(c4)
    c4 = Conv2D(filters = 128, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_42')(c4)
    c4 = BatchNormalization(name = 'conv_42_bn')(c4)
    c4 = LeakyReLU(alpha=Alpha, name = 'conv_42_act')(c4)
    c4 = Conv2D(filters = 256, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_43')(c4)
    c4 = BatchNormalization(name = 'conv_43_bn')(c4)
    c4 = LeakyReLU(alpha=Alpha, name = 'conv_43_act')(c4)       # *******___1/4 times smaller than ip___********
    c4 = MaxPooling2D(name = 'conv_4_pool')(c4)
    c4 = Dropout(dropout)(c4) 
    
    c5 = Conv2D(filters = 512, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_51')(c4)
    c5 = BatchNormalization(name = 'conv_51_bn')(c5)
    c5 = LeakyReLU(alpha=Alpha, name = 'conv_51_act')(c5)
    c5 = Conv2D(filters = 256, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_52')(c5)
    c5 = BatchNormalization(name = 'conv_52_bn')(c5)
    c5 = LeakyReLU(alpha=Alpha, name = 'conv_52_act')(c5)
    c5 = Conv2D(filters = 512, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_53')(c5)
    c5 = BatchNormalization(name = 'conv_53_bn')(c5)
    c5 = LeakyReLU(alpha=Alpha, name = 'conv_53_act')(c5)
    c5 = Conv2D(filters = 256, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_54')(c5)
    c5 = BatchNormalization(name = 'conv_54_bn')(c5)
    c5 = LeakyReLU(alpha=Alpha, name = 'conv_54_act')(c5)
    c5 = Conv2D(filters = 512, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_55')(c5)
    c5 = BatchNormalization(name = 'conv_55_bn')(c5)
    c5 = LeakyReLU(alpha=Alpha, name = 'conv_55_act')(c5)       # *******___1/4 times smaller than ip___********
    c5_p = MaxPooling2D(name = 'conv_5_pool')(c5)
    c5_p = Dropout(dropout)(c5_p) 
    
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_61')(c5_p)
    c6 = BatchNormalization(name = 'conv_61_bn')(c6)
    c6 = LeakyReLU(alpha=Alpha, name = 'conv_61_act')(c6)
    c6 = Conv2D(filters = 512, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_62')(c6)
    c6 = BatchNormalization(name = 'conv_62_bn')(c6)
    c6 = LeakyReLU(alpha=Alpha, name = 'conv_62_act')(c6)
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_63')(c6)
    c6 = BatchNormalization(name = 'conv_63_bn')(c6)
    c6 = LeakyReLU(alpha=Alpha, name = 'conv_63_act')(c6)
    c6 = Conv2D(filters = 512, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_64')(c6)
    c6 = BatchNormalization(name = 'conv_64_bn')(c6)
    c6 = LeakyReLU(alpha=Alpha, name = 'conv_64_act')(c6)
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'conv_65')(c6)
    c6 = BatchNormalization(name = 'conv_65_bn')(c6)
    c6 = LeakyReLU(alpha=Alpha, name = 'conv_65_act')(c6)       # *******___1/4 times smaller than ip___********
    c6 = Dropout(dropout)(c6)
    ################################################################
    # Layers for training model as classification
    ################################################################
    if as_classifier:
        clf = Conv2D(filters = 1000, kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'clf_1')(c6)
        clf = GlobalAveragePooling2D(name = 'pool_clf')(clf)
        clf = tf.keras.layers.Softmax(axis=-1, name = 'clf_op')(clf)
        model = Model(inputs=[input_img], outputs=[clf])
        
    ################################################################
    # Layers for training model as Detector
    # set as_classifier = False
    # set  by_name = True while loading weights
    ################################################################
    else:
        od1 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'od_1')(c6)
        od1 = BatchNormalization(name = 'od_1_bn')(od1)
        od1 = LeakyReLU(alpha=Alpha, name = 'od_1_act')(od1)
        od2 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'od_2')(od1)
        od2 = BatchNormalization(name = 'od_2_bn')(od2)
        od2 = LeakyReLU(alpha=Alpha, name = 'od_2_act')(od2)
    
        c5_t = Conv2D(filters = 64, kernel_size = (1, 1), use_bias=useBias, padding = 'same', name = 'conv_21')(c5)
        c5_t = BatchNormalization(name='skip_bn')(c5_t)
        c5_t = LeakyReLU(alpha=Alpha)(c5_t)
        c5_t = Lambda(lambda x: tf.nn.space_to_depth(x, block_size = 2), name="space_2_depth")(c5_t)
        
        od2 = Concatenate(axis = -1)([od2, c5_t])
        od3 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'od_3')(od2)
        od3 = BatchNormalization(name = 'od_3_bn')(od3)
        od3 = LeakyReLU(alpha=Alpha, name = 'od_3_act')(od3)
        od3 = Conv2D(filters = int(boxes_percell * (num_class+5)), kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'op')(od3)
        od3 = Reshape((13,13,boxes_percell, (num_class+5)), name = 'op_reshape')(od3)
        
        op = Lambda(lambda args: args[0])([od3, true_boxes])
        model = Model(inputs=[input_img, true_boxes], outputs = [op])
    
    return model  


#%%
def yolo_exception(input_img, grid_size, boxes_percell, num_class, dropout, Alpha = 0.1, batchnorm = True):
    
    
    x_model = Xception(include_top = False, weights = 'imagenet', input_shape = input_img, pooling = None, classes = num_class)#"imagenet"
    for layer in x_model.layers:
        if isinstance(layer, Activation):# putting leaky relu activation in model
            layer.activation = tf.keras.activations.relu(layer.input, alpha=Alpha)
    
    res5 = x_model.output
    # adding 4 extra conv layers
    res5 = Dropout(dropout)(res5)  
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    od3 = Conv2D(filters = 1024, kernel_size = (3, 3), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'od_3')(res5)
    od3 = BatchNormalization(name = 'od_3_bn')(od3)
    od3 = LeakyReLU(alpha=Alpha, name = 'od_3_act')(od3)
    od3 = Conv2D(filters = int(boxes_percell * (num_class+5)), kernel_size = (1, 1), use_bias=False, kernel_initializer = 'he_normal',padding = 'same', name = 'op')(od3)
    model = Model(inputs=x_model.input, outputs=[od3])
    return model










