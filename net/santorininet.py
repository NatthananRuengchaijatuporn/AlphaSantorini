import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .features import CNNFeature

class SantoriniNet():
    def __init__(self, board_dim: tuple, n_action: int, args):
        self.args = args
        self.net = Backbone(CNNFeature.n_ch, board_dim, n_action, args)
        self.action_size = n_action

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = [CNNFeature.extract(b) for b in input_boards]
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.net.model.fit(x=input_boards,
                           y=[target_pis, target_vs],
                           batch_size=self.args.batch_size,
                           epochs=self.args.epochs)

    def predict(self, board):
        # preparing input
        # feature extract
        board = CNNFeature.extract(board)
        board = board[np.newaxis, :, :, :].astype(np.float32)
        board = np.rollaxis(board, -1, 1)

        # predict
        pi, v = self.net.model.predict(board)

        return pi[0], v[0]
    
    def save(self, folder, fname):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.net.model.save_weights(os.path.join(folder, fname))

    def load(self, folder, fname):
        self.net.model.load_weights(os.path.join(folder, fname))

def getModel() :
    inp = tf.keras.Input(shape=(5,5,5))
    x1 = layers.Conv2D(128,3,1,'same')(inp)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    
    x2 = layers.Conv2D(128,3,1,'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Add()([x2,x1])
    
    x3 = layers.Conv2D(128,3,1,'same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Add()([x3,x2])
    
    x4 = layers.Conv2D(128,3,1,'same')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.Add()([x4,x3])
    
    x5 = layers.Conv2D(128,3,1,'same')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.ReLU()(x5)
    x5 = layers.Add()([x5,x4])
    
    x6 = layers.Conv2D(256,3,1,'valid')(x5)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.ReLU()(x6)
    
    x7 = layers.Conv2D(256,3,1,'same')(x6)
    x7 = layers.BatchNormalization()(x7)
    x7 = layers.ReLU()(x7)
    x7 = layers.Add()([x7,x6])
    
    x8 = layers.Conv2D(256,3,1,'same')(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.ReLU()(x8)
    x8 = layers.Add()([x8,x7])
    
    x9 = layers.Conv2D(256,3,1,'same')(x8)
    x9 = layers.BatchNormalization()(x9)
    x9 = layers.ReLU()(x9)
    x9 = layers.Add()([x9,x8])
    
    x10 = layers.Conv2D(256,3,1,'same')(x9)
    x10 = layers.BatchNormalization()(x10)
    x10 = layers.ReLU()(x10)
    x10 = layers.Add()([x10,x9])
    
    x = layers.Flatten()(x10)
    x = layers.Dense(1024,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out1 = layers.Dense(128,activation='softmax')(x)
    out2 = layers.Dense(1,activation='tanh')(x)
    model = tf.keras.Model(inputs=inp,outputs=[out1,out2])
    return model


class Backbone:
    def __init__(self, in_ch: int, board_dim: tuple, n_action: int, args):
        # game params
        self.board_x, self.board_y = board_dim
        self.action_size = n_action
        self.args = args

        n_ch = args.num_channels

        self.model = getModel()
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=tf.keras.optimizers.Adam(args.lr))
