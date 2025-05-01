from PIL import Image
from math import pow
import numpy as np
import builtins
from tqdm import tqdm
import time
import os

import Container
import Blocks


class ImageObject(object):
    def __init__(self, imageDirectory, imageName, blockDimension, outputDirectory):
        print(imageName)
        print("Step 1 of 4: Object and variable initialization")

        self.imageOutputDirectory = outputDirectory
        self.imagePath = imageName
        self.imageData = Image.open(imageDirectory + imageName)
        self.imageWidth, self.imageHeight = self.imageData.size

        if self.imageData.mode != 'L':
            self.isThisRGBImage = True
            self.imageData = self.imageData.convert('RGB')
            RGBImagePixels = self.imageData.load()
            self.imageGrayscale = self.imageData.convert('L')
            GrayscaleImagePixels = self.imageGrayscale.load()

            for yCoordinate in range(0, self.imageHeight):
                for xCoordinate in range(0, self.imageWidth):
                    redPixelValue, greenPixelValue, bluePixelValue = RGBImagePixels[xCoordinate, yCoordinate]
                    GrayscaleImagePixels[xCoordinate, yCoordinate] = int(0.299 * redPixelValue) + int(
                        0.587 * greenPixelValue) + int(0.114 * bluePixelValue)
        else:
            self.isThisRGBImage = False
            self.imageData = self.imageData.convert('L')

        self.N = self.imageWidth * self.imageHeight
        self.blockDimension = blockDimension
        self.b = self.blockDimension * self.blockDimension
        self.Nb = (self.imageWidth - self.blockDimension + 1) * (self.imageHeight - self.blockDimension + 1)
        self.Nn = 2
        self.Nf = 188
        self.Nd = 50

        self.P = (1.80, 1.80, 1.80, 0.0125, 0.0125, 0.0125, 0.0125)
        self.t1 = 2.80
        self.t2 = 0.02

        print((self.Nb, self.isThisRGBImage))

        self.featuresContainer = Container.Container()
        self.blockPairContainer = Container.Container()
        self.offsetDictionary = {}

    def run(self):
        startTimestamp = time.time()
        self.compute()
        timestampAfterComputing = time.time()
        self.sort()
        timestampAfterSorting = time.time()
        self.analyze()
        timestampAfterAnalyze = time.time()
        imageResultPath = self.reconstruct()
        timestampAfterImageCreation = time.time()

        print(("Computing time :", timestampAfterComputing - startTimestamp, "second"))
        print(("Sorting time   :", timestampAfterSorting - timestampAfterComputing, "second"))
        print(("Analyzing time :", timestampAfterAnalyze - timestampAfterSorting, "second"))
        print(("Image creation :", timestampAfterImageCreation - timestampAfterAnalyze, "second"))

        totalRunningTimeInSecond = timestampAfterImageCreation - startTimestamp
        totalMinute, totalSecond = divmod(totalRunningTimeInSecond, 60)
        totalHour, totalMinute = divmod(totalMinute, 60)
        print(("Total time    : %d:%02d:%02d second" % (totalHour, totalMinute, totalSecond), '\n'))
        return imageResultPath

    def compute(self):
        print("Step 2 of 4: Computing characteristic features")

        imageWidthOverlap = self.imageWidth - self.blockDimension
        imageHeightOverlap = self.imageHeight - self.blockDimension

        if self.isThisRGBImage:
            for i in tqdm(range(0, imageWidthOverlap + 1)):
                for j in range(0, imageHeightOverlap + 1):
                    imageBlockRGB = self.imageData.crop((i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlockGrayscale = self.imageGrayscale.crop(
                        (i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlock = Blocks.Blocks(imageBlockGrayscale, imageBlockRGB, i, j, self.blockDimension)
                    self.featuresContainer.addBlock(imageBlock.computeBlock())
        else:
            for i in range(imageWidthOverlap + 1):
                for j in range(imageHeightOverlap + 1):
                    imageBlockGrayscale = self.imageData.crop((i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlock = Blocks.Blocks(imageBlockGrayscale, None, i, j, self.blockDimension)
                    self.featuresContainer.addBlock(imageBlock.computeBlock())

    def sort(self):
        self.featuresContainer.sortFeatures()

    def analyze(self):
        print("Step 3 of 4:Pairing image blocks")
        time.sleep(0.1)
        featureContainerLength = self.featuresContainer.getLength()
        for i in tqdm(range(featureContainerLength)):
            for j in range(i + 1, featureContainerLength):
                result = self.isValid(i, j)
                if result[0]:
                    self.addDict(self.featuresContainer.container[i][0], self.featuresContainer.container[j][0],
                                 result[1])
                else:
                    break

    def isValid(self, firstBlock, secondBlock):
        if abs(firstBlock - secondBlock) < self.Nn:
            iFeature = self.featuresContainer.container[firstBlock][1]
            jFeature = self.featuresContainer.container[secondBlock][1]

            if abs(iFeature[0] - jFeature[0]) < self.P[0] and \
               abs(iFeature[1] - jFeature[1]) < self.P[1] and \
               abs(iFeature[2] - jFeature[2]) < self.P[2] and \
               abs(iFeature[3] - jFeature[3]) < self.P[3] and \
               abs(iFeature[4] - jFeature[4]) < self.P[4] and \
               abs(iFeature[5] - jFeature[5]) < self.P[5] and \
               abs(iFeature[6] - jFeature[6]) < self.P[6]:
                if abs(iFeature[0] - jFeature[0]) + abs(iFeature[1] - jFeature[1]) + abs(
                        iFeature[2] - jFeature[2]) < self.t1:
                    if abs(iFeature[3] - jFeature[3]) + abs(iFeature[4] - jFeature[4]) + abs(
                            iFeature[5] - jFeature[5]) + abs(iFeature[6] - jFeature[6]) < self.t2:
                        iCoordinate = self.featuresContainer.container[firstBlock][0]
                        jCoordinate = self.featuresContainer.container[secondBlock][0]
                        offset = (iCoordinate[0] - jCoordinate[0], iCoordinate[1] - jCoordinate[1])
                        magnitude = np.sqrt(pow(offset[0], 2) + pow(offset[1], 2))
                        if magnitude >= self.Nd:
                            return 1, offset
        return 0,

    def addDict(self, firstCoordinate, secondCoordinate, pairOffset):
        if pairOffset in self.offsetDictionary:
            self.offsetDictionary[pairOffset].append(firstCoordinate)
            self.offsetDictionary[pairOffset].append(secondCoordinate)
        else:
            self.offsetDictionary[pairOffset] = [firstCoordinate, secondCoordinate]

    def reconstruct(self):
        print("Step 4 of 4: Image reconstruction")

        groundtruthImage = np.zeros((self.imageHeight, self.imageWidth))
        linedImage = np.array(self.imageData.convert('RGB'))

        for key in sorted(self.offsetDictionary, key=lambda key: builtins.len(self.offsetDictionary[key]), reverse=True):
            if len(self.offsetDictionary[key]) < self.Nf * 2:
                break
            for i in range(len(self.offsetDictionary[key])):
                for j in range(self.offsetDictionary[key][i][1],
                               self.offsetDictionary[key][i][1] + self.blockDimension):
                    for k in range(self.offsetDictionary[key][i][0],
                                   self.offsetDictionary[key][i][0] + self.blockDimension):
                        groundtruthImage[j][k] = 255

        for x in range(2, self.imageHeight - 2):
            for y in range(2, self.imageWidth - 2):
                if groundtruthImage[x, y] == 255 and any([
                    groundtruthImage[x + dx, y + dy] == 0 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                                                         (-1, -1), (1, 1), (-1, 1), (1, -1)]
                ]):
                    linedImage[x, y, 1] = 255

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        marked_path = os.path.join(self.imageOutputDirectory, f"{timestamp}_marked_{self.imagePath}")
        bw_path = os.path.join(self.imageOutputDirectory, f"{timestamp}_bw_{self.imagePath}")

        Image.fromarray(linedImage.astype(np.uint8)).save(marked_path)
        Image.fromarray(groundtruthImage.astype(np.uint8)).save(bw_path)

        return marked_path
