
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import operator


# In[2]:

#read data,embedding array and embedding dictionary
x_data = np.load("/mnt/disk2/sayon/contents_data_embedded_format_from_600L_docs.npy")
y_data = np.load("/mnt/disk2/sayon/targets_data_embedded_format_from_600L_docs.npy")
y_data_no_pad = np.load("/mnt/disk2/sayon/targets_data_embedded_not_padded_format_from_600L_docs.npy")
embedding_array = np.load("/mnt/disk2/sayon/embedding_array_600L_docs.npy")
embedding_dict_final = np.load("/mnt/disk2/sayon/embedding_dict_final_600L_docs.npy")
x_var_length = np.load("/mnt/disk2/sayon/contents_var_length_format_from_600L_docs.npy")
y_var_length = np.load("/mnt/disk2/sayon/targets_var_length_format_from_600L_docs.npy")


# In[7]:

train_id = random.sample(range(x_data.shape[0]),int(0.9*x_data.shape[0]))


# In[8]:

valid_and_test_id = list(set(range(x_data.shape[0]))-set(list(train_id)))


# In[9]:

len(set(train_id))


# In[10]:

len(valid_and_test_id)


# In[11]:

train_id = list(train_id)


# In[12]:

valid_id = [valid_and_test_id[i] for i in random.sample(range(len(valid_and_test_id)),int(0.5*len(valid_and_test_id)))]


# In[13]:

test_id = list(set(valid_and_test_id)-set(list(valid_id)))


# In[14]:

len(set(valid_id))


# In[15]:

len(set(test_id))


# In[16]:

x_train_data = x_data[train_id]
x_train_var_length = x_var_length[train_id]
y_train_data = y_data[train_id]
y_train_var_length = y_var_length[train_id]
x_valid_data = x_data[valid_id]
x_valid_var_length = x_var_length[valid_id]
y_valid_data = y_data[valid_id]
y_valid_var_length = y_var_length[valid_id]
x_test_data = x_data[test_id]
x_test_var_length = x_var_length[test_id]
y_test_data = y_data[test_id]
y_test_var_length = y_var_length[test_id]


# In[17]:

print(x_train_data.shape,y_train_data.shape)
print(x_valid_data.shape,y_valid_data.shape)
print(x_test_data.shape,y_test_data.shape)
print len(x_train_var_length),len(y_train_var_length),len(x_valid_var_length),len(y_valid_var_length),len(x_test_var_length),len(y_valid_var_length)


# In[18]:

#creating buckets of batches for training
#each batch of min 100
#sort the training input i.e. x_train_data in ascending order of their length and arranging them
sorted_ids = np.argsort(x_train_var_length)
x_train_data_xsorted = x_train_data[sorted_ids]
x_train_var_length_xsorted = x_train_var_length[sorted_ids]
y_train_data_xsorted = y_train_data[sorted_ids]
y_train_var_length_xsorted = y_train_var_length[sorted_ids]
#now select each batch of size 100 from here in each iteration


# In[19]:

#hyperparameters
batch_size = 100
iterations = 300000
learning_rate = 0.00001
num_steps = x_train_data_xsorted.shape[1]
decoder_max_step = y_train_data_xsorted.shape[1]
num_features = embedding_array.shape[1]
num_classes = len(embedding_array)
hidden_units = num_features
final_hidden_units = 2*hidden_units
decoder_units = 2*final_hidden_units
#activation_units = 2


# In[20]:

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    #print "sequence length tf shape:",length.shape
    return length


# In[21]:

#ldecode_lengths = y_train_data_xsorted.shape[1]


# In[22]:

tf.reset_default_graph()


# In[23]:

#tensor creation for inputs, embeddings, weigths and biases
x = tf.placeholder(dtype=tf.int64,shape=[None,None])
y = tf.placeholder(dtype=tf.int64,shape=[None,None])
embeddings = tf.constant(embedding_array,dtype=tf.float32)
x_in = tf.nn.embedding_lookup(embeddings,x)
y_in = tf.nn.embedding_lookup(embeddings,y)
decoder_lengths = length(y_in)
w = {'score':tf.Variable(tf.truncated_normal(shape=[final_hidden_units,final_hidden_units]),dtype=tf.float32,name='weight_score'),
     'hdash':tf.Variable(tf.truncated_normal(shape=[decoder_units,decoder_units]),dtype=tf.float32,name='weight_hdash'),
     'decoder':tf.Variable(tf.truncated_normal(shape=[decoder_units,num_classes]),dtype=tf.float32,name='weight_decoder')
    }
b = {
     'hdash':tf.Variable(tf.constant(0.1,shape=[decoder_units,]),dtype=tf.float32,name='bias_hdash'),
     'decoder':tf.Variable(tf.constant(0.1,shape=[num_classes,]),dtype=tf.float32,name='bias_decoder')
    }


# In[24]:

#encoder cell
with tf.variable_scope('forward'):
    encoder_fw_lstm = tf.contrib.rnn.LSTMCell(hidden_units)
with tf.variable_scope('backward'):
    encoder_bw_lstm = tf.contrib.rnn.LSTMCell(hidden_units)

((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_bw=encoder_fw_lstm,
                                                              cell_fw=encoder_bw_lstm,
                                                              inputs=x_in,
                                                              dtype=tf.float32,
                                                              sequence_length=length(x_in),
                                                              time_major=False))


# In[25]:

#bidirectional step
encoder_outputs = tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c,encoder_bw_final_state.c),1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h,encoder_bw_final_state.h),1)

encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h)


# In[26]:

encoder_outputs.shape


# In[27]:

#decoder cell
decoder_lstm = tf.contrib.rnn.LSTMCell(final_hidden_units)


# In[28]:

eos_pos = embedding_dict_final.all()['<eos>']


# In[29]:

#eos tagging for decoder input
inputs = tf.placeholder(dtype=tf.int32,shape=[None,])
eos_input = tf.fill(dims=tf.shape(inputs),value=eos_pos)
eos_embedded = tf.nn.embedding_lookup(embeddings,eos_input)
new_inputs = tf.placeholder(dtype=tf.int32,shape=[None])
pad_input = tf.fill(dims=tf.shape(new_inputs),value=0)
pad_embedded = tf.nn.embedding_lookup(embeddings,pad_input)


# In[30]:

#attention based decoder
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #print initial_elements_finished
    initial_input = eos_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time,previous_output,previous_state,previous_loop_state):
    #print time
    elements_finished = (time >= decoder_lengths)
    def next_input():
        prev_out_with_weights = tf.matmul(previous_output,w['score'])
        prev_out_with_weights = tf.reshape(prev_out_with_weights,[-1,final_hidden_units,1])
        score = tf.matmul(encoder_outputs,prev_out_with_weights)
        score = tf.reshape(score,[-1,num_steps])
        attention = tf.nn.softmax(score)
        attention = tf.reshape(attention,[-1,1,num_steps])
        ct = tf.matmul(attention,encoder_outputs)
        ct = tf.reshape(ct,[-1,final_hidden_units])
        ctht = tf.concat((ct,previous_output),1)
        ht_dash = tf.nn.tanh(tf.add(tf.matmul(ctht,w['hdash']),b['hdash']))
        pred = tf.nn.softmax(tf.add(tf.matmul(ctht,w['decoder']),b['decoder']))
        prediction = tf.argmax(pred,axis=1)
        inputn = tf.nn.embedding_lookup(embeddings,prediction)
        return inputn
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(finished,lambda:pad_embedded,next_input)
    state = previous_state
    output = previous_output
    #print output.shape
    loop_state = None
    return (elements_finished,
            next_input,
            state,
            output,
            loop_state)


# In[31]:

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    #print "abc"
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


# In[32]:

#using tf.nn.raw_rnn
#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the 
#inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
#and what to emit for the output.
#ta = tensor array


# In[33]:

decoder_output_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell=decoder_lstm,loop_fn=loop_fn)


# In[34]:

decoder_outputs = decoder_output_ta.stack()


# In[35]:

decoder_outputs.shape


# In[36]:

#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))


# In[37]:

decoder_max_steps


# In[38]:

decoder_batch_size.shape


# In[39]:

decoder_dim.shape


# In[40]:

#decoder_max_steps = decoder_max_step
#decoder_dim = final_hidden_units


# In[41]:

#flattened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))


# In[42]:

decoder_outputs_flat.shape


# In[43]:

decoder_prev_out_with_weights = tf.matmul(decoder_outputs_flat,w['score'])
print decoder_prev_out_with_weights.shape,"batch_size*steps,600"
decoder_prev_out_with_weights = tf.reshape(decoder_prev_out_with_weights,[-1,decoder_max_steps,final_hidden_units])
encoder_outputs_new = tf.reshape(encoder_outputs,[-1,final_hidden_units,num_steps])
decoder_score = tf.matmul(decoder_prev_out_with_weights,encoder_outputs_new)
print "encoder_outputs_shape:",encoder_outputs_new.shape,"batch_size*600*2100"
print "decoder_outputs_shape:",decoder_prev_out_with_weights.shape,"batch_size*28*600"
print decoder_score.shape,"batch_size*28*2100"
decoder_attention = tf.nn.softmax(decoder_score)
print decoder_attention.shape,"batch_size*28*2100"
encoder_outputs_new = tf.reshape(encoder_outputs,[-1,num_steps,final_hidden_units])
decoder_ct = tf.matmul(decoder_attention,encoder_outputs_new)
print decoder_ct.shape,"batch_size*28*600"
decoder_outputs_flat_new = tf.reshape(decoder_outputs_flat,[-1,decoder_max_steps,decoder_dim])
print decoder_outputs_flat_new.shape,"batch_size,28,600"
decoder_ctht = tf.concat((decoder_ct,decoder_outputs_flat_new),2)
print decoder_ctht.shape,"batch_size,28,1200"
decoder_ctht = tf.reshape(decoder_ctht,[-1,decoder_units])
decoder_ctht_logits = tf.add(tf.matmul(decoder_ctht,w['hdash']),b['hdash'])
decoder_ctht_logits = tf.reshape(decoder_ctht_logits,[-1,decoder_max_steps,decoder_units])
print decoder_ctht_logits.shape,"batch_size,28,1200"
decoder_ht_dash = tf.nn.tanh(decoder_ctht_logits)
print decoder_ht_dash.shape,"batch_size,28,1200"
decoder_ht_dash = tf.reshape(decoder_ht_dash,[-1,decoder_units])
decoder_ht_dash_logits = tf.add(tf.matmul(decoder_ht_dash,w['decoder']),b['decoder'])
decoder_ht_dash_logits = tf.reshape(decoder_ht_dash_logits,[-1,decoder_max_steps,num_classes])
print decoder_ht_dash_logits.shape,"batch_size,28,203910"
decoder_pred = tf.nn.softmax(decoder_ht_dash_logits)
print decoder_pred.shape,"batch_size,28,203910"
decoder_pred_prediction = tf.argmax(decoder_pred,axis=2)
decoder_pred_prediction = tf.reshape(decoder_pred_prediction,[-1,decoder_max_steps])
print decoder_pred_prediction.shape,"batch_size,28"
#inputn = tf.nn.embedding_lookup(embeddings,prediction)


# In[44]:

#cross entropy loss
#one hot encode the target values so we don't rank just differentiate
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(y, depth=num_classes, dtype=tf.float32),
    logits=decoder_ht_dash_logits,
)


# In[45]:

#loss function
loss = tf.reduce_mean(stepwise_cross_entropy)


# In[46]:

#train it 
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[47]:

loss.shape


# In[54]:

x_buckets = []
x_buckets_lengths = []
y_buckets = []
y_buckets_lengths = []

len_bars = range(450,620,20)
#len_bars = len_bars + [1700, 2100, 2200]
print len_bars
bkt = []
bkt_lngts = []
y_bkt = []
y_bkt_lngts = []
j = 1
a = []

for i in range(x_train_data_xsorted.shape[0]):    
        if x_train_var_length_xsorted[i]<len_bars[j] or x_train_var_length_xsorted[i]==len_bars[j]:
            bkt.append(x_train_data_xsorted[i])
            bkt_lngts.append(x_train_var_length_xsorted[i])
            y_bkt.append(y_train_data_xsorted[i])
            y_bkt_lngts.append(y_train_var_length_xsorted[i])
            if i==(x_train_data_xsorted.shape[0]-1) :
                print len_bars[j],i
                j+=1
                x_buckets.append(np.array(bkt))
                x_buckets_lengths.append(np.array(bkt_lngts))
                y_buckets.append(np.array(y_bkt))
                y_buckets_lengths.append(np.array(y_bkt_lngts))
                bkt = []
                bkt_lngts = []
                y_bkt = []
                y_bkt_lngts = []
                break
        
        else:
            print len_bars[j],i
            j+=1
            x_buckets.append(np.array(bkt))
            x_buckets_lengths.append(np.array(bkt_lngts))
            y_buckets.append(np.array(y_bkt))
            y_buckets_lengths.append(np.array(y_bkt_lngts))
            bkt = []
            bkt_lngts = []
            y_bkt = []
            y_bkt_lngts = []
            bkt.append(x_train_data_xsorted[i])
            bkt_lngts.append(x_train_var_length_xsorted[i])
            y_bkt.append(y_train_data_xsorted[i])
            y_bkt_lngts.append(y_train_var_length_xsorted[i])


# In[59]:

k = 1
for i,j in zip(x_buckets,x_buckets_lengths):
    print k,len(i),np.max(j),np.min(j)
    k+=1


# In[64]:

number_of_batches = 800000/100


# In[65]:

number_of_batches


# In[66]:

number_of_batches_per_bucket = number_of_batches/8


# In[67]:

number_of_batches_per_bucket


# In[68]:

x_buckets = np.array(x_buckets)
x_buckets_lengths = np.array(x_buckets_lengths)
y_buckets = np.array(y_buckets)
y_buckets_lengths = np.array(y_buckets_lengths)


# In[69]:

batches_x = []
batches_y = []
batches_x_var_length = []
batches_y_var_length = []
for i in range(len(x_buckets)):
    for j in range(number_of_batches_per_bucket):
        ids = random.sample(range(len(x_buckets[i])),batch_size)
        batches_x.append(list(x_buckets[i][ids]))
        batches_y.append(list(y_buckets[i][ids]))
        batches_x_var_length.append(list(x_buckets_lengths[i][ids]))
        batches_y_var_length.append(list(x_buckets_lengths[i][ids]))
batches_x = np.array(batches_x)
batches_y = np.array(batches_y)
batches_x_var_length = np.array(batches_x_var_length)
batches_y_var_length = np.array(batches_y_var_length)    


# In[70]:

del x_buckets
del x_buckets_lengths
del y_buckets
del y_buckets_lengths
del x_train_data
del x_train_var_length
del y_train_data
del y_train_var_length


# In[71]:

print batches_x.shape
print batches_y.shape
print batches_x_var_length.shape
print batches_y_var_length.shape


# In[72]:

np.transpose(batches_y[0]).shape


# In[73]:

batches_x[0].shape


# In[74]:

embd = embedding_dict_final.all()
embd_dict = dict()
for i in embd.keys():
    embd_dict[embd[i]] = i


# In[75]:

init = tf.global_variables_initializer()


# In[ ]:

with tf.Session() as sess:
    sess.run(init)
    step = 0
    for batchx,batchy in zip(batches_x,batches_y):
        #batchx = np.reshape(batchx,[batch_size,num_steps])
        #batchy = np.transpose(batchy)
        #do,dpp, = sess.run([decoder_outputs,decoder_pred_prediction],feed_dict={x:batchx,y:batchy,inputs:batchx[:,0],new_inputs:batchx[:,0]})
        #break
        _,l = sess.run([train_op,loss],feed_dict={x:batchx,y:batchy,inputs:batchx[:,0],new_inputs:batchx[:,0]})
        if step>7990 == 0:
            lv = sess.run(loss,feed_dict={x:x_valid_data,y:y_valid_data,inputs:x_valid_data[:,0],new_inputs:x_valid_data[:,0]})
            predt = sess.run(decoder_pred_prediction,feed_dict={x:x_test_data,y:y_test_data,inputs:x_test_data[:,0],new_inputs:x_test_data[:,0]})
            print "-----At Step {0}-----".format(step)
            #print "Training Loss : ",l
            print "Validation Loss : ",lv
            #print "Testing Loss : ",lt
            print "*********Test_Output***********"
            for i in range(x_test_data.shape[0]):
                print "Full Text : ", " ".join([embd_dict[j] for j in x_test_data[i][:x_test_var_length[i]-1]])
                print "Actual Summary : ", " ".join([embd_dict[j] for j in y_test_data[i][:y_test_var_length[i]-1]])
                print "Predicted Summary : ", " ".join([embd_dict[j] for j in predt[i] if j is not 0])
                print "-------"
            print "*******************************"
	step+=1        
	print step,"steps completed", "and training loss is",l
        


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



