import tensorflow as tf
from sys import argv
from datahelper import datahelper

def main():
    dh = datahelper(argv[1])
    sess = tf.InteractiveSession()
    path = argv[2]
    new_saver = tf.train.import_meta_graph(path + '.meta')
    new_saver.restore(sess, path)
    errors = []
    errorCount = 0
    with open(argv[3], 'r') as testFile:
        for line in testFile:
            correct = True
            inputs = dh.getByName(line.strip())
            result = sess.run(tf.get_default_graph().get_tensor_by_name("finalesResult:0"),
                          feed_dict={'DefaultInput:0': inputs.data, 'DropoutRate:0': 1})
            resultList = result.tolist()
            for idx in range(len(inputs.labels)):
                if resultList[idx].index((max(resultList[idx]))) != inputs.labels[idx].index(max(inputs.labels[idx])):
                    errorCount += 1
                    correct = False
            if not correct:
                errors.append(line.strip())
    print(errorCount)
    print(errors)
    with open(argv[4], 'w') as outfile:
        outfile.write("\n".join(errors))

if __name__ == "__main__":
    main()
