import visualizations as vis
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


NUM_ITERS=200
DISPLAY_STEP=100
BATCH=100

tf.set_random_seed(0)
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)


X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10] - initialized with random values from normal distribution mean=0, stddev=0.1
W = tf.Variable(tf.truncated_normal([784, 10],stddev=0.1))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images, unroll each image row by row, create vector[784] 

XX = tf.reshape(X, [-1, 784])

# Define model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )

cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0                                                 
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualization
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])


# Initializing the variables
init = tf.global_variables_initializer()

train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)


    for i in range(NUM_ITERS+1):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(BATCH)

        if i%DISPLAY_STEP ==0:
            # compute training values for visualization
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
                        
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        # the back-propagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

title = "MNIST 1.0 single softmax layer"
vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,DISPLAY_STEP)

