import os.path
import time


class Writer:
    def __init__(self, path="log/"):
        self.__path = path

    def write(self, settings, train_accuracy, test_accuracy, losses):
        t = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        p = self.__path + t + ".txt"
        infile = open(p, "w")

        infile.write("settings:")
        for key,setting in settings.items():
            infile.write("\n")
            infile.write(key+': '+str(setting))

        infile.write("\n\ntrain accuracy:")
        for train_acc in train_accuracy:
            infile.write("\n")
            infile.write(str(train_acc))

        infile.write("\n\ntest accuracy:")
        for test_acc in test_accuracy:
            infile.write("\n")
            infile.write(str(test_acc))

        infile.write("\n\nloss:")
        for loss in losses:
            infile.write("\n")
            infile.write(str(loss))
