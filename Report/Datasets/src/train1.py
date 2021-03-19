#
# train.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

from Model1 import Model
from Vectorizer1 import Vectorizer
from Preprocessor1 import Preprocessor
import tensorflow as tf


if __name__ == "__main__":

    # PARAMETERS ---------------------------------------------------- #
    MIN_DF_VAL          = 0.1
    MAX_DF_VAL          = 0.97
    FEAT_EXTRACT_METHOD = 'count'
    LAYERS              = (512, 256, 128)
    ACTIVATION          = tf.nn.relu
    LOSS                = 'categorical_hinge'
    EPOCH               = 50
    # --------------------------------------------------------------- #

    # PREPROCESS DATA ----------------------------------------------- #
    #Create a new directory named PreprocessedDataset (takes time)
    p = Preprocessor()
    p.preprocess()
    # --------------------------------------------------------------- #

    es  = Model()
    es.read_data2()

    v = Vectorizer(max_df=MAX_DF_VAL, min_df=MIN_DF_VAL)
    v.fit(es.train_contents)
    v.save()

    train_x = v.transform(es.train_contents,    method=FEAT_EXTRACT_METHOD)
    test_x  = v.transform(es.test_contents,     method=FEAT_EXTRACT_METHOD)

    es.train_model(layers       = LAYERS,
                   tbCallBack   = [tf.keras.callbacks.TensorBoard(log_dir = 'logdir',histogram_freq=0, write_graph=True, write_images=True)],
                   train_x      = train_x,
                   train_y      = es.train_y,
                   test_x       = test_x,
                   test_y       = es.test_y,
                   loss         = LOSS,
                   activation   = ACTIVATION,
                   epoch        = EPOCH)
