# ImageRecoginitationClassificationMutationAgent
This script performs three tasks, recognizing if an image is a head shot or landscape, classifying images using K-Means and Single Link Hierarchical Clustering algorithm and finally mutating one image into other using genetic algorithm.

The script is an intelligent agent, which can classify, cluster and evolve images using kNN, K-Means, hierarchal and genetic algorithms.
The file structure before running the .py script should be as bellows:
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/parentFolder.png" />

The file structure of the images directory should be as below:
Each directory in the images folder should have corresponding images like flags folder will have all flag images, landscape folder will have all landscapes, likewise.
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/imagesFolder.png" />

Once, this is done just run the python script from command line or by double clicking on it.

Following menu will appear:
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/inputMenu.jpg">
 
Each menu item is self-explanatory and corresponding action will be taken on selecting it.
The first run of the program will be slow, since the lookup table will be created. The executions thereafter will be fast since the data will be fetched from the already available lookup table.

Below is the output of each section:
1.	Recognize my Image
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/imageRecog.png"> 

2.	Perform 3-fold cross validation
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/plot1.png">
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/plot2.png">
 
3.	K-Means Clustering
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/KMeans1.png">
 
4.	Hierarchal Clustering
Each level in hierarchal clustering depicts a merging between two clusters. The levels at the top do not contain pure clusters, they have one cluster with majority of images while other clusters with single image. The levels on at the bottom, on the other hand, has pure clusters. They have grouping of small images, like all headshots with light backgrounds grouped together or all landscapes with mountains(dark color) or landscapes with water bodies(blue color) grouped together.
 <img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/kmeans.png">
Figure 3. Overview of the page with all levels closed
<br />
<br />

<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/hier2.png">
Figure 4. Expanded Level 18  Cluster 2
<br />
<br />

5.	Hierarchal Clustering for flags data
Output will be same as the above html, again the lower levels have more purer clusters
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/hier 3.png">
<br />
6.	Evolve one image to other using genetic algorithm
The output of this is as below:
<img src="https://github.com/shantanuspark/ImageRecoginitationClassificationMutationAgent/blob/master/outputImages/mutationOutput.JPG">
 

The image to the right is the target image whereas the image to the left is the source image.
As you can see, it was able to create image somewhat similar to the input and ouput.
