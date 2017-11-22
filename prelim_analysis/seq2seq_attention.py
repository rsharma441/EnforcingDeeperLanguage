import tensorflow as tf 
from tensorflow.models.rnn.translate import data_utils
import translation as tr 


#params 
eng_size = 40000
fr_size = 40000
embedding_size = 256
en_train, fr_train, en_test, fr_test = data_utils.prepare_wmt_data('data', eng_size, fr_size)

encoder_inputs = []
decoder_inputs = []


#encoder/decoder placeholders
self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}",format(i)))


#attention
model = seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
	num_layers=1, num_units:embedding_size, num_encoder_symbols=eng_size, num_decoder_symbols=fr_size,
	embedding_size=embedding_size)

with tf.Session() as sess:
	model = create_model(sess)
	tf.train(model)