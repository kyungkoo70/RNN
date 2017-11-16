# 11/16/2017

# for RNN study
# Give three characters, predict the fourth character.


import tensorflow as tf
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

num_dic = {n: i for i, n in enumerate(char_arr)}

# print('num_dic', num_dic)

dic_len = len(num_dic)

# print('dic_len', dic_len)


# data for training
# 4-char words
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind', 'four']


def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]

        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch


learning_rate = 0.01
n_hidden = 128
total_epoch = 30

# RNN has three sequential input
n_step = 3

# input, one-hot, 26-long
n_input = n_class = dic_len

# input
# 3개가 순서대로 들어가고,
# 각각은 26-long one-hot이다.
X = tf.placeholder(dtype=tf.float32, shape=[None, n_step, n_input])

# 알파벳에 대한 index를 그대로 output으로 사용
Y = tf.placeholder(dtype=tf.int32, shape=[None])

# output으로 나오는 것,
# one-hot으로 나오는 것 같다.
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, stats = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

print('outputs-----', outputs)
outputs = tf.transpose(outputs, [1, 0, 2])
print('outputs after tranpose-----', outputs)

outputs = outputs[-1]

print('output final---->', outputs)
model = tf.matmul(outputs, W) + b

print('model', model)
print('Y', Y)

cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

# print('input_batch')
# print(input_batch)
# print('target_batch')
# print(target_batch)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch})

    print('Epoch', '%04d' % (epoch +1), 'cost = ', '{:.6f}'.format(loss))

print ('Done')

#########
# 결과 확인
######
# 레이블값이 정수이므로 예측값도 정수로 변경해줍니다.
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
# one-hot 인코딩이 아니므로 입력값을 그대로 비교합니다.
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy],
                                 feed_dict={X: input_batch, Y: target_batch})

predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)