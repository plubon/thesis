import os
import math
import numpy as np
from scipy import misc
from scipy import ndimage
import random

testCaseListName = "testCases.txt"

class datahelper:

    testProportion = 0.90

    def __init__(self, path):
        self.women = []
        self.men = []
        self.womenTest = []
        self.menTest = []
        self.menNames = []
        self.womenNames = []
        self.womenTest = []
        self.menTest = []
        self.menNamesTest = []
        self.womenNamesTest = []
        with open(path + os.sep + "labels", "r") as filelab:
            for l in filelab:
                t = l.split(";")
                dirName = path + os.sep + t[0]
                fcount = len([name for name in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, name))])
                dirName = dirName + '_1'
                fcount_1 = len([name for name in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, name))])
                if fcount > 25 or fcount_1 > 25:
                    if int(t[1]) == 0:
                        self.menNames.append(t[0])
                    elif int(t[1]) == 1:
                        self.womenNames.append(t[0])
                    else:
                        raise Exception(t[0] + " wrong label")
        random.shuffle(self.womenNames)
        random.shuffle(self.menNames)
        self.womenNamesTest = self.womenNames[int(math.ceil(len(self.womenNames) * self.testProportion)):]
        self.womenNames = self.womenNames[:int(math.ceil(len(self.womenNames)*self.testProportion))]
        self.menNamesTest = self.menNames[int(math.ceil(len(self.menNames) * self.testProportion)):]
        self.menNames = self.menNames[:int(math.ceil(len(self.menNames) * self.testProportion))]
        labelsfile = open(path + os.sep + "labels", 'r')
        for line in labelsfile:
            tokens = line.split(";")
            dirc = path+os.sep+tokens[0]
            frame_count = len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))])
            print(frame_count)
            i = 0
            while (i + 25) < frame_count:
                frames = []
                for j in range(i, i+25):
                    image = misc.imread(path + os.sep + tokens[0] + os.sep + str(j).zfill(3) + ".png")
                    sums = image.sum(axis=0).sum(axis=0)
                    if sums[2] > sums[0] or sums[2] > sums[1]:
                        raise Exception("Something wrong with channels")
                    image = np.delete(image, 2, axis=2)
                    frames.append(image)
                if tokens[0] in self.menNames:
                    self.men.append(sample(frames, [1, 0], tokens[0]))
                elif tokens[0] in self.womenNames:
                    self.women.append(sample(frames, [0, 1], tokens[0]))
                elif tokens[0] in self.menNamesTest:
                    self.menTest.append(sample(frames, [1, 0], tokens[0]))
                elif tokens[0] in self.womenNamesTest:
                    self.womenTest.append(sample(frames, [0, 1], tokens[0]))
                else:
                    raise Exception("Wrong label " + tokens[0])
                i += 5
            dirc = path + os.sep + tokens[0]+"_1"
            frame_count = len([name for name in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, name))])
            i = 0
            while (i + 25) < frame_count:
                frames = []
                for j in range(i, i + 25):
                    image = misc.imread(path + os.sep + tokens[0] + '_1' + os.sep + str(j).zfill(3) + ".png")
                    sums = image.sum(axis=0).sum(axis=0)
                    if sums[2] > sums[0] or sums[2] > sums[1]:
                        raise Exception("Something wrong with channels")
                    image = np.delete(image, 2, axis=2)
                    frames.append(image)
                if tokens[0] in self.menNames:
                    self.men.append(sample(frames, [1, 0], tokens[0]+'_1'))
                elif tokens[0] in self.womenNames:
                    self.women.append(sample(frames, [0, 1], tokens[0]+'_1'))
                elif tokens[0] in self.menNamesTest:
                    self.menTest.append(sample(frames, [1, 0], tokens[0]+'_1'))
                elif tokens[0] in self.womenNamesTest:
                    self.womenTest.append(sample(frames, [0, 1], tokens[0]+'_1'))
                else:
                    raise Exception("Wrong label " + tokens[0] + '_1')
                i += 5
        self.proportion = len(self.men) / float(len(self.women) + len(self.men))
        print("Proportion: " + str(self.proportion))
        print("No of men samples: " + str(len(self.men)))
        print("No of men test samples: " + str(len(self.menTest)))
        print("No of women samples: " + str(len(self.women)))
        print("No of women test samples: " + str(len(self.womenTest)))
        with open(testCaseListName, 'w') as outfile:
            outfile.write("\n".join(self.menNamesTest))
            outfile.write("\n".join(self.womenNamesTest))

    def get_label(self, val):
        if int(val) == 1:
            return [0, 1]
        elif int(val) == 0:
            return [1, 0]
        else:
            raise Exception("Wrong label")


    def getByName(self, name):
        l1 = [x for x in self.men if x.name == name]
        l2 = [x for x in self.menTest if x.name == name]
        l3 = [x for x in self.women if x.name == name]
        l4 = [x for x in self.womenTest if x.name == name]
        l11 = [x for x in self.men if x.name == name+'_1']
        l21 = [x for x in self.menTest if x.name == name+'_1']
        l31 = [x for x in self.women if x.name == name+'_1']
        l41 = [x for x in self.womenTest if x.name == name+'_1']
        lf = l1 + l2 + l3 + l4 + l11 + l21 + l31 + l41
        if len(lf) == 0:
            print("Empty for " + name)
        arrays = []
        labels = []
        for item in lf:
            arrays.append(item.img)
            labels.append(item.label)
        return batch(np.stack(arrays, 0), labels)

    def getnextbatch(self, size):
        men_indices = random.sample(range(len(self.men)), int(math.ceil(self.proportion * size)))
        women_indices = random.sample(range(len(self.women)), int(math.floor((1 - self.proportion) * size)))
        listsum = []
        arrays = []
        labels = []
        for idx in men_indices:
            listsum.append(self.men[idx])
        for idx in women_indices:
            listsum.append(self.women[idx])
        random.shuffle(listsum)
        for item in listsum:
            arrays.append(item.img)
            labels.append(item.label)
        return batch(np.stack(arrays, 0), labels)

    def getsingledata(self):
        arrays = [self.men[0].img]
        labels = [self.men[0].label]
        return batch(np.stack(arrays, 0), labels)
            

    def getlldata(self):
        ret = self.men + self.women + self.menTest + self.womenTest
        for item in ret:
            item.img = np.stack([self.img], 0)
        return ret

    def gettestdata(self):
        arrays = []
        labels = []
        names = []
        listsum = self.menTest + self.womenTest
        random.shuffle(listsum)
        for item in listsum:
            arrays.append(item.img)
            labels.append(item.label)
            names.append(item.name)
        subarrays = [arrays[x:x + 100] for x in xrange(0, len(arrays), 100)]
        sublabels = [labels[x:x+100] for x in xrange(0, len(labels), 100)]
        subnames = [names[x:x+100] for x in xrange(0, len(names), 100)]
        ret = []
        for i in range(len(subarrays)):
            ret.append(batch(np.stack(subarrays[i], 0), sublabels[i], subnames[i]))
        return ret


class sample:
    def __init__(self, img, label, name):
        self.img = np.concatenate(img, 2)
        self.label = label
        self.name = name


class batch:
    def __init__(self, data, labels, names=[]):
        self.data = data
        self.labels = labels
        self.names = names

