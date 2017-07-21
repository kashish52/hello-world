import numpy as np
import random
import _pickle as cPickle
import gzip
from PIL import Image
import scipy.io as sc
import os

import cv2
# from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive


class Neuralnetwork(object):
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        i = 0

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b

            a = self.sigmoid(z)

        return a

    def Stochastic(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(n):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))


            else:
                print("Epoch {0} complete".format(j))

            if j == epochs:
                break;

    def update_mini_batch(self, mini_batch, eta):

        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(b.shape) for b in self.weights]

        for x, y in mini_batch:
            schange_b, schange_w = self.backprop(x, y)

            change_b = [cb + scb for cb, scb in zip(change_b, schange_b)]
            change_w = [cw + scw for cw, scw in zip(change_w, schange_w)]

        self.biases = [b - (eta / len(mini_batch)) * cb for b, cb in zip(self.biases, change_b)]
        self.weights = [w - (eta / len(mini_batch)) * cw for w, cw in zip(self.weights, change_w)]

    def backprop(self, x, y):

        schangew = [np.zeros(b.shape) for b in self.weights]
        schangeb = [np.zeros(b.shape) for b in self.biases]

        # feedforward

        activation = x  # first layer activation that is input only
        activationlist = [x]  # to store all activation to find error
        zs = []  # to store all z for sigmoid prime

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activationlist.append(activation)

        # backward pass using 4 bp equations

        delta = self.cost_derivative(activationlist[-1], y) * self.sigmoiddash(z[-1])
        schangeb[-1] = delta
        schangew[-1] = np.dot(delta, activationlist[-2].transpose())

        # find rest of the delta using 2nd  equation of bp


        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoiddash(z)
            schangeb[-l] = delta
            schangew[-l] = np.dot(delta, activationlist[-l - 1].transpose())

        return (schangeb, schangew)

    def cost_derivative(self, a, y):
        return a - y

    def sigmoiddash(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def savewandb(self):

        return (self.weights, self.biases)

    def setwnandb(self, weights, biases):

        self.weights = weights
        self.biases = biases


class Training_data(object):
    def load_data(self):

        global content7, content6
        content7 = []

        q = 0
        with open("new dataset.txt") as f:
            content2 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content2 = [x.strip() for x in content2]

        with open("value new data.txt") as f:
            content3 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content3 = [x.strip() for x in content3]
        content = []
        content1 = []
        for x, y in zip(content2, content3):
            if (int(y) > 9 and int(y) < 36):
                content.append(x)
                content1.append(int(y) - 10)

        with open("new dataset.txt") as f:
            content5 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content5 = [x.strip() for x in content5]

        with open("value new data.txt") as f:
            content4 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content4 = [x.strip() for x in content4]
        content2 = []
        content3 = []
        for x, y in zip(content5, content4):
            if (int(y) > 9 and int(y) < 36):
                content2.append(x)
                content3.append(int(y) - 10)

        test_data = []
        training_data = []
        t_values = []
        t1_values = []

        print(type(content), len(content1), len(content2), len(content3))

        for i, p in zip(content, content1):
            e = np.zeros((784, 1))
            e1 = np.zeros((26, 1))
            l = i.split()
            for j in range(0, len(l)):
                e[j] = float(l[j]) / 255.0

            e1[int(p)] = 1.0
            training_data.append(e)
            t_values.append(e1)

        print("hello")

        for i, p in zip(content2, content3):
            e = np.zeros((784, 1))

            l = i.split()
            for j in range(0, len(l)):
                e[j] = float(l[j]) / 255.0

            test_data.append(e)
            t1_values.append(int(p))

        #print(len(content2))
        training_data = list(zip(training_data, t_values))
        test_data = list(zip(test_data, t1_values))

        with open("traintval.txt") as f:
            content6 = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content6 = [x.strip() for x in content6]
        content7 = []

        for item in content6:
            e = np.zeros((26, 1))
            e[int(item)] = 1.0
            content7.append(e)

        return (training_data, test_data)


class segmentation(object):
    def hprojections(self, imgarr):

        x = imgarr.shape
        imgarr1 = (imgarr / 255.0 * 0.99) + 0.01
        ncount = []

        t = x[1] - 1.1
        n = 0

        for i in range(0, x[0]):

            sumr = 0
            n = n + 1

            for j in range(0, x[1]):
                sumr = sumr + imgarr1[i][j]

            # print(sumr)

            if sumr > t:
                ncount.append(n)

        return (ncount)

    def vprojections(self, imgarr):

        x = imgarr.shape
        imgarr1 = (imgarr / 255.0 * 0.99) + 0.01
        ncount = []
        ncount.append(1)

        t = x[0] - 1.1
        n = 0

        for j in range(0, x[1]):

            sumc = 0
            n = n + 1

            for i in range(0, x[0]):
                sumc = sumc + imgarr1[i][j]

            # print(sumc)
            if (sumc > t):
                # print(s)
                ncount.append(n)
        ncount.append(x[1])

        return (ncount)

    def charsegmentation(self, rword):

        global rwchar, t_data

        rwchar = []
        t_data = []

        for j in rword:
            rwchar2 = []

            for i in j:
                #print(type(i))
                rwchar1 = []

                imgarr = i
                x = imgarr.shape
                ncount1 = []
                diff = []
                ncount = self.vprojections(imgarr)

                for i in range(0, (len(ncount) - 1)):
                    if (ncount[i + 1] - ncount[i] > 1):
                        ncount1.append(ncount[i])
                        diff.append(ncount[i + 1] - ncount[i])
                #print(ncount)
                #print(ncount1)
                #print(diff)
                h = x[0]

                for k in range(0, len(diff)):

                    w = diff[k]
                    start = ncount1[k] - 1

                    data = np.zeros((h, w), dtype=np.uint8)
                    for i in range(0, h):

                        for j in range(0, w):
                            data[i][j] = imgarr[i][start + j]

                    data = np.invert(data)

                    dim = (28, 28)

                    # perform the actual resizing of the image and show it
                    data = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)
                    data1 = data / 255.0
                    rwchar1.append(data)
                    t_data.append(data1.reshape((784, 1)))

                    im = Image.fromarray(data)
                    im.show()

                rwchar2.append(rwchar1)
            rwchar.append(rwchar2)

    def wordsegmentation(self, row):

        rword = []

        for i in range(0, len(row), 2):

            imgarr = row[i]
            word = []
            x = imgarr.shape

            ncount1 = []
            diff = []
            ncount = self.vprojections(imgarr)
            # print(ncount)
            ws = 0
            scount = []
            scount1 = []
            wcount = []
            for i in range(0, (len(ncount) - 1)):
                ws = ws + 1
                if (ncount[i + 1] - ncount[i] > 1):
                    wcount.append(ws)
                    scount.append(ncount[i])
                    scount1.append(ncount[i + 1])
                    ws = 0

            wcount.append(ws)

            # print(wcount)
            # print(scount)
            # print(scount1)
            sumdiff = 0
            for i in wcount:
                sumdiff += i

            avg = sumdiff / len(wcount)

            start = 0
            end = 0
            sum = 0
            # print(avg)

            diff.append(scount[0])

            for i in range(0, len(wcount)):

                if wcount[i] > avg:
                    if i != 0 and i != (len(wcount) - 1):
                        ncount1.append(scount1[i - 1])
                    if i < len(scount) and i != 0:
                        diff.append(scount[i])

            ncount1.append(scount1[len(scount1) - 1])
            # ncount1.append(scount1[0])
            # print(diff)

            # print(ncount1)

            for k in range(0, len(ncount1)):
                h = x[0]
                w = ncount1[k] - diff[k]
                # print(diff[k])
                data = np.zeros((h, w), dtype=np.uint8)
                for i in range(0, x[0]):
                    for j in range(0, w):
                        data[i][j] = imgarr[i][diff[k] + j]

                im = Image.fromarray(data)
                # im.show()
                word.append(data)

            rword.append(word)

        #print(len(rword))
        self.charsegmentation(rword)

    def linesegmentation(self, imgarr):

        ncount = self.hprojections(imgarr)
        x = imgarr.shape
        ncount1 = []
        diff = []
        row = []
        for i in range(0, (len(ncount) - 1)):
            if (ncount[i + 1] - ncount[i] > 1):
                ncount1.append(ncount[i])
                diff.append(ncount[i + 1] - ncount[i])

        # print(ncount)
        # print(ncount1)
        # print(diff)
        w = x[1]

        for k in range(0, len(diff)):

            h = diff[k]
            start = ncount1[k] - 1

            data = np.zeros((h, w), dtype=np.uint8)
            for i in range(0, h):

                for j in range(0, w):
                    data[i][j] = imgarr[start + i][j]

            im = Image.fromarray(data)
            # im.show()
            row.append(data)
            row.append('/n')
        # im = Image.fromarray(row[0])
        # im.save("test.bmp")
        # r1 = []
        # r1.append(row[0])
        # print(len(r1))
        self.wordsegmentation(row)


class thresholding(object):
    def thresh(self, image):
        ratio = image.shape[0] / 500.0
        orig = image.copy()
        warped = imutils.resize(image, height=500)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = threshold_adaptive(warped, 251, offset=10)
        warped = warped.astype("uint8") * 255
        gray = cv2.GaussianBlur(warped, (5, 5), 0)
        # show the original and scanned images
        # print ("STEP 3: Apply perspective transform")
        print("hello")
        dst = cv2.fastNlMeansDenoising(warped, None, 10, 5, 21)
        im = Image.fromarray(imutils.resize(dst, height=650))
        # im.show()

        return (dst)


def callproject(path):
    #path = soc.givepath()
    print(path)
    img = cv2.imread(path)
    
    img = cv2.fastNlMeansDenoising(img, None, 5, 11, 25)
    im = thresholding()
    img = im.thresh(img)

    imgarr = np.array(img)

    test = segmentation()

    test.linesegmentation(imgarr)

    # print(len(rwchar))








    # d=Training_data()

    # training_data,test_data=d.load_data()
    # t_data=list(zip(t_data,content6))
    # im1=test_data[234][0]
    # print(test_data[234][1])

    # ex=test_data[234][0]
    # ex=ex*255.0
    # ex=ex.reshape((28,28))
    # ex=ex.transpose()
    # im=Image.fromarray(ex)
    # im.show()

    net = Neuralnetwork([784, 50, 26])

    # net.Stochastic(training_data, 30, 10, 3.0,test_data)

    # net.SGD(t_data, 25, 10, 3.0)

    # weights,biases=net.savewandb()



    # np.save("weights.npy",weights)
    # np.save("biases.npy",biases)

    b = np.load('weights.npy')
    a = np.load('biases.npy')
    net.setwnandb(b, a)

    # print(np.argmax(net.feedforward(test_data[234][0])))






    result = ""
    result1 = []

    for row in rwchar:

        for word in row:

            for char in word:
                inp = char / 255.0

                inp = inp.transpose()

                inp = inp.reshape((784, 1))
                nchar = np.argmax(net.feedforward(inp))
                #print(nchar)

                char = chr(65 + nchar)

                result = result + char

            result = result+" "

        result=result + "\n"


    # print(result)
    #print(result)

    with open("hello.txt", "w") as text_file:
        text_file.write(result)
    result1="TQTI IS OYQ GECTDTD IMDTE"



    os.system('open hello1.txt')
    return result






class jsocket(object):
    def __init__(self):
        import socket  # Import socket module
        self.path = ""
        soc = socket.socket()  # Create a socket object
        host = "localhost"  # Get local machine name
        port = 47896  # Reserve a port for your service.
        soc.bind((host, port))  # Bind to the port
        print("hello")
        soc.listen(5)  # Now wait for client connection.
        # print(len("Hello kashish suneja"))
        while True:
            conn, addr = soc.accept()  # Establish connection with client.
            print("Got connection from", addr)
            msg = conn.recv(1024)
            # print(len(msg))
            # print(msg)
            break
        self.path = self.path + str(msg[2:])
        self.path = self.path[2:len(self.path) - 1]
        sendstr="this is the result to be printed"

        sendstr=callproject(self.path)
        print(self.path)
        conn.send(sendstr.encode())
        soc.close()

    def givepath(self):
        return (self.path)


soc = jsocket()



