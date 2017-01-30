import layers
import tensorflow as tf
from datahelper import *
import logging
import time

class network:

    reportFrequency = 50

    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.dropoutRate = tf.placeholder(tf.float32, name="DropoutRate")
        self.session = None
        self.dataHelper = None
        self.layers = []
        self.layers.append(layers.conv(7, 96, 50, 1))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.conv(5, 192, 96, 2))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.conv(3, 512, 192, 1))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.conv(2, 4096, 512, 1))
        self.layers.append(layers.dense(3*4096, 4096, dropout_rate=self.dropoutRate, reshape_needed=True))
        self.layers.append(layers.dense(4096, 2048, dropout_rate=self.dropoutRate))
        self.layers.append(layers.dense(2048, 2, name="FinalResult"))
        self.input = tf.placeholder(tf.float32, [None, 60, 40, 50], name="DefaultInput")
        self.normalizedInput = tf.truediv(tf.sub(self.input, tf.constant(128.)), tf.constant(128.), name = "NormalizedInput")
        self.results = []
        self.results.append(self.layers[0].result(self.normalizedInput))
        for i in xrange(1, len(self.layers)):
            try:
                self.results.append(self.layers[i].result(self.results[i-1]))
            except:
                print(i)
                raise
        self.finalResult = self.results[len(self.results) - 1]
        self.reallyFinalResult = tf.identity(self.finalResult, name="finalesResult")
        print(self.reallyFinalResult.get_shape())
        self.labels = tf.placeholder(tf.float32, [None, 2], name="Labels")
        self.crossEntropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.finalResult, self.labels))
        self.train_step = tf.train.AdamOptimizer(epsilon=0.001, learning_rate=0.0001).minimize(self.crossEntropy, global_step=self.global_step)
        self.correct_prediction = tf.equal(tf.argmax(self.finalResult, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="Accuracy")
        self.saver = None

    def test(self, path):
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        self.dataHelper = datahelper(path)
        data = self.dataHelper.getsingledata()
        print(data.data.shape)
        print(len(data.labels))
        res = self.results[len(self.results)-1].eval(feed_dict={
            self.input: data.data, self.labels: data.labels, self.dropoutRate: 0.5})
        print("finshed, output %g", res)

    def train(self, path, epochs, batchsize):
        counter = 0
        maxAcc = 0.
        saver = tf.train.Saver()
        self.costs = []
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.dataHelper = datahelper(path)
        print("started")
        logging.basicConfig(filename="logs" + os.sep + time.ctime() + '.log', level=logging.DEBUG)
        logging.info("epochs: "+str(epochs))
        logging.info("batch size: "+str(batchsize))
        logging.info("test data proportion: "+str(1 - datahelper.testProportion))
        logging.info("Started at" + time.ctime())
        for i in xrange(epochs):
            newbatch = self.dataHelper.getnextbatch(batchsize)
            if i % network.reportFrequency == 0 and i > 0:
                results = self.session.run([self.accuracy, self.crossEntropy],feed_dict={self.input: newbatch.data, self.labels: newbatch.labels, self.dropoutRate: 1})
                self.costs.append(results[1])
                print(results)
            self.train_step.run(feed_dict={self.input: newbatch.data, self.labels: newbatch.labels, self.dropoutRate: 0.6})
        logging.info("Finished training at" + time.ctime())
        testdata = self.dataHelper.gettestdata()
        finAcc = 0
        test_len = 0
        correctMen = 0
        correctWomen = 0
        totalMen = 0
        totalWomen = 0
        for batch in testdata:
            acc = self.session.run([self. accuracy, self.reallyFinalResult], feed_dict={
                self.input: batch.data, self.labels: batch.labels, self.dropoutRate: 1})
            finAcc += acc[0] * len(batch.data)
            resultList = acc[1].tolist()
            for idx in range(len(batch.labels)):
                if batch.labels[idx][0] == 1:
                    totalMen += 1
                    if resultList[idx].index((max(resultList[idx]))) == batch.labels[idx].index(max(batch.labels[idx])):
                        correctMen += 1
                else:
                    totalWomen += 1
                    if resultList[idx].index((max(resultList[idx]))) == batch.labels[idx].index(max(batch.labels[idx])):
                        correctWomen += 1
            test_len += len(batch.data)
        finAcc = finAcc / test_len
        print(finAcc)
        name = time.ctime()
        if not os.path.exists('models' + os.sep + name):
            os.makedirs('models' + os.sep + name)
        logging.info("Total women in test set: " + str(totalWomen))
        logging.info("Total men in test set: " + str(totalMen))
        logging.info("Correcly classified women: " + str(correctWomen))
        logging.info("Correctly classifed men: " + str(correctMen))
        saver.save(self.session, 'models' + os.sep + name + os.sep + 'my-model')
        logging.info("final accuracy %g" % finAcc)
        logging.info("Finished run at" + time.ctime())
        costString = "\n"
        for cost in self.costs:
            costString += str(cost)+"\n"
        logging.info("costs: "+costString)




