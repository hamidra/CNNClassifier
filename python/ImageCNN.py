from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
class ImageCNN:
  def __init__(self, dataPicklefile):
    self.pickle_file = dataPicklefile
    self.image_size = 640
    self.num_labels = 2
    self.num_channels = 3 # grayscale
    
    self.batch_size = 2
    self.patch_size = 5
    self.depth = 16
    self.num_hidden = 64
    self.num_steps = 1001

  @staticmethod
  def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
  
  def reformat(self, dataset, labels):
    dataset = dataset.reshape(
      (-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
    labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

  def open_data_set(self, pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    self.train_dataset, self.train_labels = self.reformat(train_dataset, train_labels)
    self.valid_dataset, self.valid_labels = self.reformat(valid_dataset, valid_labels)
    self.test_dataset, self.test_labels = self.reformat(test_dataset, test_labels)
    print('Training set', self.train_dataset.shape, self.train_labels.shape)
    print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
    print('Test set', self.test_dataset.shape, self.test_labels.shape)
    return {"dataset": self.train_dataset, "label": self.train_labels}, {"dataset": self.valid_dataset, "label": self.valid_labels}, {"dataset": self.test_dataset, "label": self.test_labels}

  def get_Graph(self):    
    
    return graph


  def runSession(self, num_steps, outDirectory=""):
    graph = tf.Graph()
    
    with graph.as_default():
      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [self.patch_size, self.patch_size, self.num_channels, self.depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([self.depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [self.patch_size, self.patch_size, self.depth, self.depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [self.image_size // 4 * self.image_size // 4 * self.depth, self.num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [self.num_hidden, self.num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))
      
      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
      
      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))

    with tf.Session(graph=graph) as session:
      saver = tf.train.Saver()
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
        batch_data = self.train_dataset[offset:(offset + self.batch_size), :, :, :]
        batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % ImageCNN.accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % ImageCNN.accuracy(
            valid_prediction.eval(), self.valid_labels))
      print('Test accuracy: %.1f%%' % ImageCNN.accuracy(test_prediction.eval(), self.test_labels))
      
      if outDirectory!="":
        print('model is saved as %s' % saver.save(session, outDirectory))

def main():
  datafile  = input("Eneter the path to the dataset pickel file to load the data: ") 
  outPath   = input("Eneter the path to out directory to save the model: ")
  classifier = ImageCNN(datafile)
  classifier.open_data_set(classifier.pickle_file)
  classifier.runSession(300, outPath)

if __name__=='__main__':
  main()
  