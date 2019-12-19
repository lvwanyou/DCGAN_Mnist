import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from load_normal_data import DataLoad

#1，添加了 9-12行
#2，修改了所有的784为24
#3，修改了122行的X_mb为real_data
#4，注释掉了113-121行
normal_data = DataLoad(64)#batchsize
BATCHNO = 16
mb_size = 64#batchsize保持一致
Z_dim = 100
CLIP = [-0.01, 0.01] #权重裁剪

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 24])

D_W1 = tf.Variable(xavier_init([24, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 24]))
G_b2 = tf.Variable(tf.zeros(shape=[24]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2

    return D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


G_sample = generator(Z)
D_logit_real = discriminator(X)
D_logit_fake = discriminator(G_sample)

#把此处的注释掉，下面的打开即可
print("shape1..")
print(D_logit_real.shape)
print("shape2.....")
print(D_logit_fake.shape)

# D_loss = -tf.reduce_mean(tf.log(D_logit_real) + tf.log(1. - D_logit_fake))
# G_loss = -tf.reduce_mean(tf.log(D_logit_fake))
#注释掉即为不去log，log的先取后取，没有影响的所以再算mean之前取也可以的
D_loss = -tf.reduce_mean(D_logit_real + 1. - D_logit_fake)
G_loss = -tf.reduce_mean(D_logit_fake)

# Alternative losses:
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)#lr构造方法默认是0.001
# G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(D_loss, var_list=theta_D)#lr构造方法默认是0.001
G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(G_loss, var_list=theta_G)
clip_d_op = [var.assign(tf.clip_by_value(var,CLIP[0],CLIP[1])) for var in theta_D]#对判器的权重矩阵进行裁剪

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('outd/'):
    os.makedirs('outd/')



    # fname, X, fmt = '%.18e', delimiter = ' ', newline = '\n', header = '',
    # footer = '', comments = '# ', encoding = None
    # fig = plot(samples)
    # plt.savefig('outd/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    # i += 1
    # plt.close(fig)

    #X_mb, _ = mnist.train.next_batch(mb_size)
for it in range(50):#这个是epoch
    normal_data.create_batches("modbus_write_single_register.txt")
    real_data = normal_data.next_batch()
    #print("x_mb........")
    #print(X_mb)
    #np.savetxt("test.txt",X_mb[0])

    #每个epoch输出生成的测试用例
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
    print("source.....")
    print(samples)
    print("after deal with...")
    last = np.round(np.multiply(samples, 15.0))

    np.savetxt("newtest_line.txt", last[0], fmt="%d", delimiter=" ", newline=" ")
    np.savetxt(fname='newtest_all.txt', X=last, fmt='%d', delimiter=" ", newline="\n")

    with open('datasix.txt', 'a') as fout:
        for arr in last.astype('int32'):
            bu = ' '.join([str(x) for x in arr]) + '\n'
            fout.write(bu)


    for _ in range(BATCHNO):#这个是训练的一个epoch需要的次数即iteration
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: real_data, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    #每个epoch写日志
    log = open('experiment-log.txt', 'a')  # 写日志
    print('Iter: {}'.format(it))
    print('D loss: {:.4}'.format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print()
    log.write('Iter: {}'.format(it))
    log.write("\n")
    log.write('D loss: {:.4}'.format(D_loss_curr))
    log.write("\n")
    log.write('G_loss: {:.4}'.format(G_loss_curr))
    log.write("\n")
    log.close()
