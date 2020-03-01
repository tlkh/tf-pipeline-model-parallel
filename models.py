import tensorflow as tf
from tensorflow.keras import layers
import nvtx.plugins.tf as nvtx_tf
from nvtx.plugins.tf.keras.layers import NVTXStart, NVTXEnd

def cnn_block():
    cnn_block = tf.keras.models.Sequential([
        layers.Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform"),
        layers.Conv2D(512, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform"),
        layers.Conv2D(512, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform"),
        layers.Conv2D(128, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform"),
        layers.BatchNormalization(fused=True),
    ])
    return cnn_block


class ParallelCNN(tf.keras.Model):
    """
    Normal model parallel CNN
    """
    def __init__(self, num_classes=10):
        super(ParallelCNN, self).__init__()
        with tf.device('/GPU:0'):
            self.conv_0 = layers.Conv2D(32, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform")
            self.maxpool_1 = layers.MaxPooling2D((2,2))
            self.block_1 = cnn_block()
        with tf.device('/GPU:1'):
            self.block_2 = cnn_block()
            self.block_3 = cnn_block()
            self.block_4 = cnn_block()
            self.maxpool_2 = layers.MaxPooling2D((2,2))
            self.flat = layers.Flatten()
            self.classifier = layers.Dense(num_classes, activation="softmax")
            
    def forward_1(self, split_batch):
        with tf.device('/GPU:0'):
            x = self.conv_0(split_batch)
            x = self.maxpool_1(x)
            ret = self.block_1(x)
            return ret
    
    def forward_2(self, split_batch):
        with tf.device('/GPU:1'):
            x = self.block_2(split_batch)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.maxpool_2(x)
            x = self.flat(x)
            ret = self.classifier(x)
            return ret

    def call(self, inputs):
        inputs, nvtx_context = nvtx_tf.ops.start(inputs, message='forward_1',
                                                 domain_name='forward', grad_domain_name='grad')
        x = self.forward_1(inputs)
        x = nvtx_tf.ops.end(x, nvtx_context)
        x, nvtx_context = nvtx_tf.ops.start(x, message='forward_2',
                                            domain_name='forward', grad_domain_name='grad')
        ret = self.forward_2(x)
        ret = nvtx_tf.ops.end(ret, nvtx_context)
        return ret
        


class PipelineCNN(tf.keras.Model):
    """
    Pipeline model parallel CNN
    """
    def __init__(self, splits=2, num_classes=10):
        super(PipelineCNN, self).__init__()
        self.splits = splits
        with tf.device('/GPU:0'):
            self.conv_0 = layers.Conv2D(32, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform")
            self.maxpool_1 = layers.MaxPooling2D((2,2))
            self.block_1 = cnn_block()
        with tf.device('/GPU:1'):
            self.block_2 = cnn_block()
            self.block_3 = cnn_block()
            self.block_4 = cnn_block()
            self.maxpool_2 = layers.MaxPooling2D((2,2))
            self.flat = layers.Flatten()
            self.classifier = layers.Dense(num_classes, activation="softmax")
            
    def forward_1(self, split_batch):
        with tf.device('/GPU:0'):
            x = self.conv_0(split_batch)
            x = self.maxpool_1(x)
            x = self.block_1(x)
            return x
    
    def forward_2(self, split_batch):
        with tf.device('/GPU:1'):
            x = self.block_2(split_batch)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.maxpool_2(x)
            x = self.flat(x)
            x = self.classifier(x)
            return x

    def call(self, inputs):
        with tf.device('/GPU:0'):
            splits = tf.split(inputs, self.splits, axis=0, num=self.splits, name="split_batch")
        
        # pipe 1
        
        pipe_1 = splits[0]
        
        pipe_1, nvtx_context_p1 = nvtx_tf.ops.start(pipe_1, message='pipe_1',
                                                    domain_name='forward', grad_domain_name='grad')
        
        pipe_1, nvtx_context_p1f1 = nvtx_tf.ops.start(pipe_1, message='p1_forward_1',
                                                      domain_name='forward', grad_domain_name='grad')
        pipe_1 = self.forward_1(pipe_1)
        pipe_1 = nvtx_tf.ops.end(pipe_1, nvtx_context_p1f1)
        
        pipe_1, nvtx_context_p1f2 = nvtx_tf.ops.start(pipe_1, message='p1_forward_2',
                                                      domain_name='forward', grad_domain_name='grad')
        pipe_1 = self.forward_2(pipe_1)
        pipe_1 = nvtx_tf.ops.end(pipe_1, nvtx_context_p1f2)
        
        pipe_1 = nvtx_tf.ops.end(pipe_1, nvtx_context_p1)
        
         # pipe 2
        
        pipe_2 = splits[1]
        
        pipe_2, nvtx_context_p2 = nvtx_tf.ops.start(pipe_2, message='pipe_2',
                                                 domain_name='forward', grad_domain_name='grad')
        
        pipe_2, nvtx_context_p2f1 = nvtx_tf.ops.start(pipe_2, message='p2_forward_1',
                                                      domain_name='forward', grad_domain_name='grad')
        pipe_2 = self.forward_1(pipe_2)
        pipe_2 = nvtx_tf.ops.end(pipe_2, nvtx_context_p2f1)
        
        pipe_2, nvtx_context_p2f2 = nvtx_tf.ops.start(pipe_2, message='p2_forward_2',
                                                      domain_name='forward', grad_domain_name='grad')
        pipe_2 = self.forward_2(pipe_2)
        pipe_2 = nvtx_tf.ops.end(pipe_2, nvtx_context_p2f2)
        
        pipe_2 = nvtx_tf.ops.end(pipe_2, nvtx_context_p2)
        
        with tf.device('/GPU:1'):
            ret = tf.concat([pipe_1, pipe_2], 0, name="concat_batch")
            return ret
        
        
class SingleCNN(tf.keras.Model):
    """
    Non-model parallel CNN
    """
    def __init__(self, num_classes=10):
        super(SingleCNN, self).__init__()
        with tf.device('/GPU:0'):
            self.conv_0 = layers.Conv2D(32, (3,3), padding="same", activation="relu", kernel_initializer="he_uniform")
            self.maxpool_1 = layers.MaxPooling2D((2,2))
            self.block_1 = cnn_block()
            self.block_2 = cnn_block()
            self.block_3 = cnn_block()
            self.block_4 = cnn_block()
            self.maxpool_2 = layers.MaxPooling2D((2,2))
            self.flat = layers.Flatten()
            self.classifier = layers.Dense(num_classes, activation="softmax")
            
    def forward_1(self, split_batch):
        with tf.device('/GPU:0'):
            x = self.conv_0(split_batch)
            x = self.maxpool_1(x)
            ret = self.block_1(x)
            return ret
    
    def forward_2(self, split_batch):
        with tf.device('/GPU:0'):
            x = self.block_2(split_batch)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.maxpool_2(x)
            x = self.flat(x)
            ret = self.classifier(x)
            return ret

    def call(self, inputs):
        x = self.forward_1(inputs)
        ret = self.forward_2(x)
        return ret
    
    