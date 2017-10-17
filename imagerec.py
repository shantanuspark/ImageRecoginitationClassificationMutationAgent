try:
    from PIL import Image
    import numpy as np
    import operator
    import pickle
    from random import randint,shuffle 
    from matplotlib import pyplot as plt 
    import os
    import webbrowser
except:
    print "Required libraries not installed, quitting.."
    exit()

class ImageRepr(object):
    '''
    Custom Image Class
    '''
    def __init__(self,colorValues):
        self.colorAttributeVector = []
        for key in colorValues:
            self.colorAttributeVector.append(colorValues[key])
    
def euclidianDistance(row1,row2):
    '''
    Calculates euclidian distance between 2 rows
    '''
    distance = 0
    for i in range(len(row1)-1):
        distance += pow((float(row1[i]) - float(row2[i])), 2)
    return distance

'''
Have referred to https://www.youtube.com/watch?v=IqfPGcNStE8 tutorial
'''
def extractImageAttributes(image):
    imageArray = np.asarray(image)
    colorValues = {}
    colorValues["r1"] = 0
    colorValues["r2"] = 0
    colorValues["r3"] = 0
    colorValues["r4"] = 0
    colorValues["g1"] = 0
    colorValues["g2"] = 0
    colorValues["g3"] = 0
    colorValues["g4"] = 0
    colorValues["b1"] = 0
    colorValues["b2"] = 0
    colorValues["b3"] = 0
    colorValues["b4"] = 0
    pixelCount = 0
    for eachRow in imageArray:
        for eachPixel in eachRow:
            if eachPixel[0] <64:
                colorValues["r1"]+=1
            elif eachPixel[0] <128:
                colorValues["r2"]+=1
            elif eachPixel[0] <192:
                colorValues["r3"]+=1
            else:
                colorValues["r4"]+=1
                
            if eachPixel[1] <64:
                colorValues["g1"]+=1
            elif eachPixel[1] <128:
                colorValues["g2"]+=1
            elif eachPixel[1] <192:
                colorValues["g3"]+=1
            else:
                colorValues["g4"]+=1
                
            if eachPixel[2] <64:
                colorValues["b1"]+=1
            elif eachPixel[2] <128:
                colorValues["b2"]+=1
            elif eachPixel[2] <192:
                colorValues["b3"]+=1
            else:
                colorValues["b4"]+=1
            pixelCount+=1
    
    for key in colorValues.keys():
        colorValues[key]/=float(pixelCount)   
    return ImageRepr(colorValues)

def vectorAddition(vector1, vector2):
    '''
    Adds vector 1 and vector 2
    '''
    resultVector = vector1
    for index in range(len(vector2)):
        resultVector[index]+=vector2[index]
    
    return resultVector

def divideVector(vector,divider):
    '''
    Divides vector by the provided natural number(divider)
    '''
    for key in range(len(vector)):
        vector[key]/=float(divider)
    return vector

def calculateTermination(centroids):
    '''
    Find the terminating condition by comparing distance of existing centroid with its previous iteration counterpart
    ''' 
    distance = 0
    itr = 0
    for key in centroids.keys():
        if itr == 2:
            break
        #calculate sum of distances of each centroid with their older counterparts in the previous iteration
        distance += euclidianDistance(centroids[key], centroids[key+2])
        itr+=1
    
    #getting the mean distance of movement of each centroid
    distance /= 2
    
    if distance < 0.01:
        return True
    else:
        return False

class Agent(object):
    '''
    kNN image recoginition agent
    '''
    images = []
    testImage = ""
    k = 1
    def sensor(self, images, testImage, k):
        self.images = images
        self.testImage = testImage
        self.k = k
        return self.function()
    
    def function(self): 
        for image in self.images:
            image.distance = euclidianDistance(image.colorAttributeVector, self.testImage.colorAttributeVector)
        
        return self.actuator() 
    
    def actuator(self):
        return sorted(self.images,key=operator.attrgetter('distance'))[0:self.k]
            

class Environment(object):
    folders = ["headshot","landscape"]
    images = []
    
    def readNCreateFeatures(self):
        '''
        Reads training images and create custom representaions of them
        '''
        try:
            with open('lookupTable.pickle', 'rb') as handle:
                self.images = pickle.load(handle)  
        except:
            print '\nBuilding up lookup table, first run of the program will take some time, please be patient!'
            for folder in self.folders:
                print 'Extracting attributes of',folder,'images..'
                #files = next(os.walk('.'))[2]
                imageNames = next(os.walk('./images/'+folder))[2]
                for imageName in imageNames:
                    path = 'images/'+folder+'/'+imageName
                    image = Image.open(path)
                    imageRepr = extractImageAttributes(image)
                    imageRepr.path = path
                    imageRepr.type = folder
                    self.images.append(imageRepr)
                print folder,'images processed'
            with open('lookupTable.pickle', 'wb') as handle:
                    pickle.dump(self.images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print 'Saved the lookup table, next runs of the program will be faster\n'
                
    def predictImageType(self, k, testImage):
        print "Indentifying your image, this may take some time.."
        results = Agent().sensor(self.images, extractImageAttributes(testImage), k)
        headshot = 0
        for result in results:
            if result.type == 'headshot':
                headshot+=1
        print "***********************************************************"
        if headshot > k/2:
            print "Image is a headshot"
        else:
            print "Image is a landscape"
        print "***********************************************************"
    
    '''
    Generates training and validation sets, performs validation and plots accuracy vs k graph
    '''        
    def performCrossValidation(self, k):
        
        folds = {}
        folds[0] = []
        folds[1] = []
        folds[2] = []
        
        lIndex=0
        hIndex=0
        for image in self.images:
            if image.type == 'headshot':
                folds[hIndex%3].append(image)
                hIndex+=1
            else:
                folds[lIndex%3].append(image)
                lIndex+=1    
        
        accuracies = {}
        foldAccuracies = []
        x = range(1,k+1)
        
        #Compute accuracy of each k
        for tempK in x:
            fold0Accuracy = self.validate(folds[2] + (folds[1]), folds[0], tempK)
            foldAccuracies.append(fold0Accuracy)
            fold1Accuracy = self.validate(folds[2]+(folds[0]), folds[1], tempK)
            foldAccuracies.append(fold0Accuracy)
            fold2Accuracy = self.validate(folds[1]+(folds[0]), folds[2], tempK)
            foldAccuracies.append(fold0Accuracy)
            accuracies[tempK] = (fold0Accuracy+fold1Accuracy+fold2Accuracy)/float(3)
        
        self.drawGraph(x, accuracies.values(), "Average fold Accuracy vs k plot", "3-Fold Cross Validation", "value of k", "Accuracy")
        
        self.drawGraph(range(1,len(foldAccuracies)+1), foldAccuracies, "Fold accuracy vs k plot", "3-Fold Cross Validation", "Folds for each value of k", "Accuracy")
        
        
    '''
    Validates the validationSet against trainingSet for a given value of k and returns accuracy
    '''    
    def validate(self, trainingSet, validationSet, k):
        for image in validationSet:
            results = Agent().sensor(trainingSet, image, k)
            headshot = 0;
            for result in results:
                if result.type == 'headshot':
                    headshot+=1
            if headshot > k/2:
                image.type = 'headshot'
            else:
                image.type = 'landscape'
        correctPrediction = 0
        itr = 0
        for image in validationSet:
            if image.type == self.images[itr].type:
                correctPrediction+=1
            itr+=1
        
        return float(correctPrediction)/len(validationSet)
    
    '''
    Plots graph as per the information provided in parameters
    '''
    def drawGraph(self, x, y, title, supTitle, xlabel, ylabel):
        plt.suptitle(supTitle)
        plt.title(title)
        plt.xlabel(xlabel) 
        plt.ylabel(ylabel) 
        plt.xlim([1,len(x)])
        plt.plot(x,y,'b', markersize=15)
        plt.grid(True) 
        plt.show()
    
    '''
    Clusters images using K-Means Clustring and prints its accuracy
    '''    
    def clusterWithKMeans(self):
        shuffledImages = self.images
        
        #shuffling images
        shuffle(shuffledImages)
        
        centroids = {}
        tempIndexes = []
        resultImages = []
        clusterTypes = {}
        
        #randomly select initial centroids
        for index in range(2):
            while True:
                imageIndex = randint(0, len(shuffledImages)-1)
                if not tempIndexes.__contains__(imageIndex):
                    if len(tempIndexes)>0 and shuffledImages[tempIndexes[0]].type == shuffledImages[imageIndex].type:
                        continue
                    break
            tempIndexes.append(imageIndex)
            centroids[index] = shuffledImages[imageIndex].colorAttributeVector
            clusterTypes[index] = shuffledImages[imageIndex].type
        
        print "\nPerforming K-Means clustering on",len(shuffledImages),"images"
        itr = 0
        while True:
            
            #Find distance of each image from the centroid
            for image in shuffledImages:
                image.distFromcentroid0 = euclidianDistance(centroids[itr*2+0], image.colorAttributeVector)
                image.distFromcentroid1 = euclidianDistance(centroids[itr*2+1], image.colorAttributeVector)
            
            itr+=1

            #Assign the closest centroid as the centroid for each image
            for image in shuffledImages:
                if image.distFromcentroid0 <= image.distFromcentroid1:
                    image.centroid = 0
                else:
                    image.centroid = 1  
            
            resultImages = sorted(shuffledImages,key=operator.attrgetter('centroid'))
            
            #Find mean vector for each centroid group and add new centroid
            centroidCount = {}
            
            for image in resultImages:
                try:
                    #compute vector addition and count of rows in each vector
                    centroids[2*itr+image.centroid] = vectorAddition(image.colorAttributeVector, centroids[2*itr+image.centroid])
                    centroidCount[2*itr+image.centroid] += 1
                except KeyError:
                    centroids[2*itr+image.centroid] = image.colorAttributeVector  
                    centroidCount[2*itr+image.centroid] = 1  
            
            #get mean by dividing vector addition of all jobs of a centroid by count of all jobs of that centroid
            for centroidId in centroidCount:
                centroids[centroidId] = divideVector(centroids[centroidId],centroidCount[centroidId])
            
            #evaluate the termination condition
            if calculateTermination(centroids):
                break
            
            #remove old centroids          
            i = 0
            for key in centroids.keys():
                if i==2:
                    break
                i+=1
                centroids.pop(key)
                
            #maximize distance of old centroids in each job
            for image in resultImages:
                image.distFromcentroid0 = 9
                image.distFromcentroid1 = 9
                
        correctPrediction = 0
        for key in clusterTypes.keys():
            for image in resultImages:
                # checking if the image is correctly clustered
                if image.centroid%2 == key and image.type == clusterTypes[key]:
                    correctPrediction+=1
        
        accuracy = round(correctPrediction/float(len(resultImages)),2)
        print "***********************************************************"        
        print "Finished K-Means Clustering with an accuracy of",accuracy,"!!"
        print "***********************************************************"
        htmlOutputer(resultImages, accuracy, "K-Means", clusterTypes)
        return resultImages
    
    '''
    Clusters images using single link hierarchal clustering and prints its accuracy
    '''
    def clusterWithSingleLink(self, isFlags=False, imagesToBeClustered=[]):
        
        if not isFlags:
            imagesToBeClustered = self.images
            shuffle(imagesToBeClustered)
        
        print "\n***********************************************************"
        print "Kindly be patient, hirearchal clustering will take time.."
        
        itr = len(imagesToBeClustered)
        minI = 0
        minJ = 0
        dist = {}
        for i in range(len(imagesToBeClustered)):
            dist[i] = {}
            imagesToBeClustered[i].group = {}
            
        while itr>0:
            print "Processed",len(imagesToBeClustered)-itr,"out of",len(imagesToBeClustered)
            minDist = 9999999999
            for i in range(len(imagesToBeClustered)):
                for j in range(len(imagesToBeClustered)):
                    #find's minimum(single link) euclidean distance between ith and jth image clusters in itr'th level
                    if i==j:
                        continue
                    if itr == len(imagesToBeClustered):
                        dist[i][j] = euclidianDistance(imagesToBeClustered[i].colorAttributeVector,imagesToBeClustered[j].colorAttributeVector)
                    else:
                        dist[i][j] = self.findMinDist(imagesToBeClustered[i].group[itr+1],imagesToBeClustered[j].group[itr+1], dist)
                    if minDist > dist[i][j]:
                        minI = i
                        minJ = j
                        minDist = dist[i][j]
                        
                #For the first iteration closest images will have <i,j> populated with itr as key 
            if itr == len(imagesToBeClustered):
                for i in range(len(imagesToBeClustered)):
                    if i == minI:
                        #image's itr attribute holds the grouped images
                        imagesToBeClustered[i].group[itr] = [i,minJ]
                    elif i == minJ:
                        imagesToBeClustered[i].group[itr] = [minI,i]
                    else:
                        imagesToBeClustered[i].group[itr] = [i]
                #For other iteration's itr will be appended with previous itr's and the minimum of current iteration
            else: 
                #copy previous itr group to the current group
                imagesToBeClustered[minI].group[itr] = list(imagesToBeClustered[minI].group[itr+1])
                #grouping rows with minimum distances
                for index in imagesToBeClustered[minJ].group[itr+1]:
                    if not imagesToBeClustered[minI].group[itr].__contains__(index):
                        imagesToBeClustered[minI].group[itr].append(index)
                imagesToBeClustered[minJ].group[itr] = list(imagesToBeClustered[minJ].group[itr+1])
                for index in imagesToBeClustered[minI].group[itr+1]:
                    if not imagesToBeClustered[minJ].group[itr].__contains__(index):
                        imagesToBeClustered[minJ].group[itr].append(index)
                #Add level group for each member of the group except for minI and minJ
                for index in range(len(imagesToBeClustered)):
                    if index==minI or index==minJ:
                        continue
                    imagesToBeClustered[index].group[itr] = list(imagesToBeClustered[index].group[itr+1])
                #Add missing elements of each group
                for image in imagesToBeClustered[minI].group[itr]:
                    if image==minI or image==minJ:
                        continue
                    for index in imagesToBeClustered[minI].group[itr]:
                        if not imagesToBeClustered[image].group[itr].__contains__(index):
                            imagesToBeClustered[image].group[itr].append(index)
            
            itr-=1
        
        for index in range(1, len(imagesToBeClustered)):
            for image in imagesToBeClustered:
                image.group[index] = sorted(image.group[index])
        
        levelClustersDict = {}
        for level in range(1,len(imagesToBeClustered)+1):
            uniqueClusters = []
            for image in imagesToBeClustered:
                if not uniqueClusters.__contains__(image.group[level]):
                    uniqueClusters.append(image.group[level])
            levelClustersDict[level] = uniqueClusters
            
        htmlOutputerForSingleLink(imagesToBeClustered, levelClustersDict)
            
        return imagesToBeClustered
            
    '''
    Finds single link(min) distance between two clusters
    '''
    def findMinDist(self, clusterI, clusterJ, distMatrix):
        minDist = 9999999999
        if sorted(clusterI) == sorted(clusterJ):
            return minDist
        for i in clusterI:
            for j in clusterJ:
                if i==j:
                    continue
                dist = distMatrix[i][j]
                if dist < minDist:
                    minDist = dist
        return minDist     


def htmlOutputerForSingleLink(resultImages, levelClustersDict):
    '''
    Outputs the job results in the html format
    '''

    outputFile = open("output.html","w")
    htmlContent = '''
            <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <title>Intelligent Search Agent</title>
    <!-- Bootstrap -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css" integrity="sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M" crossorigin="anonymous">

    
    </head>
    <body>
        <div class="container mb-10">
            <div class="mt-2">
                <h3>
                  Image Recognition and Clustering
                  <small class="text-muted">Intelligent Agent</small>
                </h3>
            </div>
            
            <div class="alert alert-info" role="alert">Similar Images Grouped together using <strong>Single Link Hierarchical</strong> Clustering Algorithm</div>
    '''
    levelData = ""
    for level in levelClustersDict.keys():
        clusterData = ""
        i=0
        for cluster in levelClustersDict[level]:
            imgSrc = ""
            for index in cluster:
                imgSrc+="<img src='"+resultImages[index].path+"' class='img-thumbnail'>"
            i+=1
            clusterData += '''
        <div id="accordionTwo'''+repr(i)+'''" role="tablist">
          
          <div class="card mt-1 " >
            <div class="card-header" role="tab" id="headingOne">
              <h6 class="mb-0">
                <a data-toggle="collapse" href="#'''+repr(level)+repr(i)+'''" aria-expanded="true" aria-controls="collapseOne">
                        Cluster '''+repr(i)+''' <span class="badge badge-pill badge-success">'''+repr(len(cluster))+'''</span> 
                </a>
              </h6>
            </div>
        
            <div id="'''+repr(level)+repr(i)+'''" class="collapse " role="tabpanel" aria-labelledby="headingOne" data-parent="#accordionTwo">
            
              <div class="card-body">
                  '''+imgSrc+'''
              </div>
              
            </div>
          </div>
          
         
          
        </div>
        
        '''
        levelData += '''
    <div id="accordion" role="tablist" class="mt-2">
      <div class="card">
        <div class="card-header" role="tab" id="headingOne">
          <h6 class="mb-0">
            <a data-toggle="collapse" href="#level'''+repr(level)+'''" aria-expanded="true" aria-controls="collapseOne">
                   Level '''+repr(level)+''' <span class="badge badge-pill badge-success">'''+repr(len(levelClustersDict[level]))+'''</span> 
            </a>
          </h6>
        </div>
    
        <div id="level'''+repr(level)+'''" class="collapse" role="tabpanel" aria-labelledby="headingOne" data-parent="#accordion">
        
          <div class="card-body ">
          '''+clusterData+'''
          
          </div>
      
    </div>
  </div>
    
</div> '''
    
    footer='''

<div class="alert alert-secondary text-center mt-5" role="alert">Go back to the python console to continue performing other image operations</div>

</div>
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js" integrity="sha384-b/U6ypiBEHpOf/4+1nzFpr53nxSS+GLCkfwBdFNTxtclqqenISfwAzpKaMNFNmj4" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/js/bootstrap.min.js" integrity="sha384-h0AbiXch4ZDo7tp9hKZ4TsHbi047NrKGLO3SEJAg45jXxnGIfYzk4Si90RDIqNm1" crossorigin="anonymous"></script>
</body>
</html>
    '''
    outputFile.write(htmlContent+levelData+footer)
    webbrowser.open_new_tab('file://' + os.path.realpath("output.html")) 
    print "\nKindly open url -> "+'file://' + os.path.realpath("output.html") + " in browser to view results\n"
    outputFile.close()


def htmlOutputer(resultImages, accuracy, algoName, clusterTypes):
    '''
    Outputs the job results in the html format
    '''
    imgSrc = {}
    imgSrc['headshot'] = ""
    imgSrc['landscape'] = ""
    for image in resultImages:
        imgSrc[clusterTypes[image.centroid]]+="<img src='"+image.path+"' class='img-thumbnail'>"
        
    outputFile = open("output.html","w")
    htmlContent = '''
            <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <title>Intelligent Search Agent</title>
    <!-- Bootstrap -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css" integrity="sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M" crossorigin="anonymous">
    
    </head>
    <body>
        <div class="container mb-10">
            <div class="mt-2">
                <h3>
                  Image Recognition and Clustering
                  <small class="text-muted">Intelligent Agent</small>
                </h3>
            </div>
            
            <div class="alert alert-info" role="alert">Similar Images Grouped together using <strong>'''+algoName+'''</strong> Clustering Algorithm with an accuracy of <span class="badge badge-info">'''+str(accuracy)+'''</span></div>
    
    <div id="accordion" role="tablist">
      <div class="card">
        <div class="card-header" role="tab" id="headingOne">
          <h6 class="mb-0">
            <a data-toggle="collapse" href="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                    '''+imgSrc.keys()[0]+''' Cluster 
            </a>
          </h6>
        </div>
    
        <div id="collapseOne" class="collapse mx-auto" role="tabpanel" aria-labelledby="headingOne" data-parent="#accordion">
        
          <div class="card-body ">
          '''+imgSrc[imgSrc.keys()[0]]+'''
                </div>
      
    </div>
  </div>
  
  <div class="card mt-1">
    <div class="card-header" role="tab" id="headingTwo">
      <h6 class="mb-0">
        <a class="collapsed" data-toggle="collapse" href="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
          '''+imgSrc.keys()[1]+''' Cluster
        </a>
      </h6>
    </div>
    <div id="collapseTwo" class="collapse mx-auto" role="tabpanel" aria-labelledby="headingTwo" data-parent="#accordion">
      <div class="card-body">
         '''+imgSrc[imgSrc.keys()[1]]+'''      
      </div>
    </div>
  </div>
  
</div>

<div class="alert alert-secondary text-center mt-5" role="alert">Go back to the python console to continue performing other image operations</div>

</div>
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js" integrity="sha384-b/U6ypiBEHpOf/4+1nzFpr53nxSS+GLCkfwBdFNTxtclqqenISfwAzpKaMNFNmj4" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/js/bootstrap.min.js" integrity="sha384-h0AbiXch4ZDo7tp9hKZ4TsHbi047NrKGLO3SEJAg45jXxnGIfYzk4Si90RDIqNm1" crossorigin="anonymous"></script>
</body>
</html>
    '''
    outputFile.write(htmlContent)
    webbrowser.open_new_tab('file://' + os.path.realpath("output.html")) 
    print "\nKindly open url -> "+'file://' + os.path.realpath("output.html") + " in browser to view results\n"
    
    outputFile.close() 


def readNCreateFeaturesOfImages():
    '''
    Reads training images and create custom representaions of them
    '''
    flags = next(os.walk('./images/flags'))[2]
    flagImageRepr = []
    for flag in flags:
        path = "./images/flags/"+flag
        image = Image.open(path)
        imageRepr = extractImageAttributes(image)
        imageRepr.path = path
        flagImageRepr.append(imageRepr)
    
    return flagImageRepr

    '''
    print 'Extracting attributes of Flags..'
    for fileName in range(1,51):
        path = 'images/flags/'+repr(fileName)+'.jpg'
        image = Image.open(path)
        imageRepr = extractImageAttributes(image)
        imageRepr.path = path
        imageRepr.type = folder
        self.images.append(imageRepr)

        print 'Saved the lookup table, next runs of the program will be faster\n'
    '''
    

   
def acceptInputK(isCrossFoldValidation=False):
    while True:
        try:
            if not isCrossFoldValidation:
                k = int(raw_input("Enter value of k\n>>> "))
            else:
                k = int(raw_input("Enter maximum value of k, you want me to analyze till..\n>>> "))
            if k>100:
                raise AttributeError()
            return k
        except AttributeError:
            print "WARNING : Value of k is large, computing will take a lot of time, kindly consider reducing the value of k"
            if 'c'==raw_input("Press c to continue with the existing value of k\nPress any other key to renter a new value for k\n>>>"):
                return k
        except ValueError:
            print "ERROR : Wrong input encountered, kindly enter the correct value for k.."


'''
Returns a chromosome row of length lenChromosome
'''
def createChromosomeRow(lenChromosome):
    chromosomeRow = []
    i = 0
    while i < lenChromosome:
        pixel = []
        pixel.append(randint(0,256))
        pixel.append(randint(0,256))
        pixel.append(randint(0,256))
        chromosomeRow.append(pixel)
        i+=1
    return chromosomeRow

'''
Creates chromosome population of 'number' size with each population having 'chromosomeCount' chromosomes
'''
def createChromosomePopulation(number, chromosomeCount):
    population = []
    i = 0
    while i < number:
        population.append(createChromosomeRow(chromosomeCount))
        i+=1
    return population


def manhattanDistance(row1,row2):
    '''
    Calculates manhattan distance between 2 rows
    '''
    distance = 0
    for i in range(len(row1)):
        for j in range(len(row1[i])):
            distance += abs(float(row1[i][j]) - float(row2[i][j]))
    return distance

'''
Get top 2 fittest chromosome by calculating manahattan distance between each chromosome in population and target chromosome
'''
def getFittestChromosomes(population, targetRow):
    minVals = [99999999, 99999999]
    minRows = [population[0],population[1]]
    for row in population:
        distance = manhattanDistance(row, targetRow)
        if distance < minVals[0]:
            minRows[0] = row
            minVals[0] = distance
        elif distance < minVals[1]:
            minRows[1] = row
            minVals[1] = distance
    return minRows


def mutate(chromosome1, chromosome2):
    population = []
    #Gradually increasing the cross over value till the length of chromosome
    for i in range(1, len(chromosome2)-1):
        childChromosome = []
        childChromosome.extend(chromosome1[0:i])
        childChromosome.extend(chromosome2[i:len(chromosome2)])
        population.append(np.asarray(childChromosome))
    #adding just chromosome2's genes in new chromosome and getting new values for 1st part
    for i in range(1, len(chromosome2)-1):
        childChromosome = []
        childChromosome.extend(createChromosomeRow(i))
        childChromosome.extend(chromosome2[i:len(chromosome2)])
        population.append(np.asarray(childChromosome))
    #adding chromosome 2 before chromosome 1
    for i in range(1, len(chromosome2)-1):
        childChromosome = []
        childChromosome.extend(chromosome2[0:i])
        childChromosome.extend(chromosome1[i:len(chromosome2)])
        population.append(np.asarray(childChromosome))
    #adding only chromosome1's genes in new chromosome getting rest from chromoxome 1
    for i in range(1, len(chromosome2)-1):
        childChromosome = []
        childChromosome.extend(createChromosomeRow(i))
        childChromosome.extend(chromosome1[i:len(chromosome2)])
        population.append(np.asarray(childChromosome))
    
    return population
        
    
class GeneticAlgorithmAgent():
    def sensor(self,sourceImage,destinationImage):
        print "Starting Genetic Algorithm, this process may take time, please be patient.."
        self.targetImage = Image.open(destinationImage)
        self.sourceImage = Image.open(sourceImage).resize(self.targetImage.size,Image.ANTIALIAS)
        self.function()
    
    '''
    Have referred http://www.ai-junkie.com/ga/intro/gat2.html to understand the intricacies of genetic algorithm
    '''
    def function(self):
        sourceArray = np.asarray(self.sourceImage)
        targetArray = np.asarray(self.targetImage)
        finalArray = []
        
        rowIndex = 0
        chromosomePopulation = []
        for row  in targetArray:
            print "Genetically mutating to the find best fit for row",rowIndex,"out of",len(row)
            #create initial population
            chromosomePopulation = createChromosomePopulation(100, len(row))
            #append the first row of source image to the population
            chromosomePopulation.append(sourceArray[rowIndex])
            itr = 0
            while True:
                #get the fittest chromosome
                fittestChromosomes = getFittestChromosomes(chromosomePopulation, row)
                #mutate fittest chromosomes to create new population
                chromosomePopulation = mutate(fittestChromosomes[0],fittestChromosomes[1])
                itr+=1
                
                if manhattanDistance(row, fittestChromosomes[0]) < 15000 or itr>15:
                    finalArray.append(fittestChromosomes[0])
                    break
            rowIndex+=1
        self.finalArray = finalArray
        self.sourceArray = sourceArray
        self.targetArray = targetArray
        self.action()
    
    def action(self):        
        #convert the final array to int8
        finalArray = np.asarray(self.finalArray).astype(np.uint8)
        #print the finally created image
        fig1 = plt.figure()
        fig1.suptitle("Mutated Image")
        plt.imshow(finalArray,aspect="auto")
        fig2 = plt.figure()
        fig2.suptitle("Source Image")
        plt.imshow(self.sourceArray)
        fig3 = plt.figure()
        fig3.suptitle("Target Image")
        plt.imshow(self.targetArray)
        plt.show()

            
print "Welcome to the Intelligent Image Recognition Agent!! \n"
env = Environment()
env.readNCreateFeatures()

while True:
    try:
        searchKey = raw_input("Action Menu: \n1. Recognize my Image\n2. Perform 3-fold Cross Validation\n3. Cluster Images using K-Means Algorithm\n4. Cluster Images using Single link hierarchical clustering Algorithm\n5. Cluster Flags using hierarchical clustering\n6. Evolve one flag to another using Genetic Algorithm\n7. Delete lookup table\n8. Exit\n>>> ")
        if int(searchKey) == 1:
            files = next(os.walk('.'))[2]
            inputImages = {}
            i = 0
            for imFile in files:
                if imFile.__contains__('.jpg'):
                    i+=1
                    inputImages[i]=imFile
            inputImageText = "Which image should I recognize? (To get your image listed in the below list, please add it the same directory as the .py script[Only .jpg supported])"
            for key in inputImages.keys():
                inputImageText+="\n"+repr(key)+". "+inputImages[key]
            inputImageText+="\n>>> "
            inputImage = -1
            k = 1
            while True:
                try:
                    inputImage = int(raw_input(inputImageText))
                    if not inputImages.has_key(inputImage):
                        raise ValueError()
                    break
                except ValueError:
                    print "ERROR : Either you entered a wrong input or no jpg images are in the same directory as the .py script.."    
                    #if 'b'==raw_input("Press b to go back to the previous menu\nPress any other key to renter above options\n"):
                    #   raise ValueError()
            k = acceptInputK()
            i = Image.open(inputImages[inputImage])
            env.predictImageType(k,i)
        elif int(searchKey) == 2:
            env.performCrossValidation(acceptInputK(True))  
        elif int(searchKey) == 3:
            env.clusterWithKMeans()
        elif int(searchKey) == 4:
            env.clusterWithSingleLink()
        elif int(searchKey) == 5:
            env.clusterWithSingleLink(True,readNCreateFeaturesOfImages())
        elif int(searchKey) == 6:
            files = next(os.walk('.'))[2]
            inputImages = {}
            i = 0
            for imFile in files:
                if imFile.__contains__('.jpg'):
                    i+=1
                    inputImages[i]=imFile
            inputImageText = "Which is the source Image? (To get your image listed in the below list, please add it the same directory as the .py script[Only .jpg supported])"
            for key in inputImages.keys():
                inputImageText+="\n"+repr(key)+". "+inputImages[key]
            inputImageText+="\n>>> "
            inputImage = -1
            outputImage = -1
            k = 1
            while True:
                try:
                    inputImage = int(raw_input(inputImageText))
                    if not inputImages.has_key(inputImage):
                        raise ValueError()
                    break
                except ValueError:
                    print "ERROR : Either you entered a wrong input or no jpg images are in the same directory as the .py script.." 
            while True:
                try:
                    outputImage = int(raw_input(inputImageText.replace("source", "target")))
                    if not inputImages.has_key(outputImage) or outputImage==inputImage:
                        raise ValueError()
                    break
                except ValueError:
                    print "ERROR : Either you entered a wrong input(source image cannot be the same as target) or no jpg images are in the same directory as the .py script.."  
                    
            GeneticAlgorithmAgent().sensor(inputImages[inputImage], inputImages[outputImage])
            
        elif int(searchKey) == 7:
            os.remove('lookupTable.pickle')   
            print "Lookup table deleted..!"
        elif int(searchKey) == 8:
            print "Thank you!"
            exit()
        else:
            raise ValueError()
    except ValueError:
        print "ERROR : Wrong input encountered, kindly enter the correct option.."