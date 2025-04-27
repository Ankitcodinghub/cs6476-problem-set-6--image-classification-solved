# cs6476-problem-set-6--image-classification-solved
**TO GET THIS SOLUTION VISIT:** [CS6476 Problem Set 6- Image Classification Solved](https://www.ankitcodinghub.com/product/cs6476-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;123847&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS6476 Problem Set 6- Image Classification  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
Problem Set 6: Image Classification

ASSIGNMENT DESCRIPTION

Description

In this problem set you will be implementing face recognition using PCA, (Ada)Boosting, and the Viola-Jones algorithm.

Learning Objectives

• Learn the advantages and disadvantages of PCA, Boosting, and Viola-Jones.

• Learn how face detection/recognition work, explore the pros &amp; cons of various techniques applied to this problem area.

• Identify the inherent challenges of working on these face detection methods.

Problem Overview

Methods to be used

In this assignment you will be implementing PCA, Boosting, and HaarFeatures algorithms from scratch. Unlike previous problem sets, you will be coding them without using OpenCV functions dedicated to solve the problem.

Please do not use absolute paths in your submission code. All paths should be relative to the submission directory. Any submissions with absolute paths are in danger of receiving a penalty!

INSTRUCTIONS

Programming Instructions

Your main programming task is to complete the api described in the file ps6.py. The driver program experiment.py helps to illustrate the intended use and will output the files needed for the writeup. Additionally there is a file ps6_test.py that you can use to test your implementation.

Note for this assignment, the numba library is required for efficient code runtime. The library is just included for the helper classes and you will not be required to learn the library to use it. Simply use ’pip install numba’ to install it.

Write-up Instructions

Create ps6_report.pdf – a PDF file that shows all your output for the problem set, including images labeled appropriately (by filename, e.g. ps6-1-a-1.png) so it is clear which section they are for and the small number of written responses necessary to answer some of the questions (as indicated). For a guide as to how to showcase your results, please refer to the latex template for PS4.

How to Submit

Two assignments have been created on Gradescope: one for the report – PS6_report, and the other for the code – PS6_code.

• Report: the report (PDF only) must be submitted to the PS6_report assignment.

• Code: all files must be submitted to the PS6_code assignment. DO NOT upload zipped folders or any sub-folders, please upload each file individually. Drag and drop all files into Gradescope.

Notes

• If you wish to modify the autograder functions, create a copy of those functions and DO NOT mess with the original function call.

1. PCA [30 POINTS]

Principalcomponentanalysis(PCA)isatechniquethatconvertsasetofattributesintoasmaller set of attributes, thus reducing dimensionality. Applying PCA to a dataset of face images generates eigenvalues and eigenvectors, in this context called eigenfaces. The generated eigenfaces can be used to represent any of the original faces. By only keeping the eigenfaces with the largest eigenvalues (and accordingly reducing the dimensionality) we can quickly perform face recognition.

In this part we will use PCA to identify people in face images. We will use the Yalefaces dataset, and will try to recognize each individual person.

1.a. Loading images

Loading images: As part of learning how to process this type of input, you will complete the function load_images. Read each image in the images_files variable, resize it to the dimensions provided, and create a data structure X that contains all resized images. Each row of this array is a flattened image (see np.flatten).

You will also need the labels each image belongs to. Each image file has been named using the following format:subject##.xyz.png. We will use the number in ## as our label (“01” -&gt; 1, “02” -&gt; 2, etc.).

CreatealistoflabelsthatmatcheachimageinXusingthefilenamestrings. Next,usetheXarray to obtain the mean face (µ) by averaging each column. You will then use the resulting structure to reshape it to a 2D array and later save it as an image. Complete visualize_mean_face in experiment.py to produce images for the report.

Code:

-load_images(folder, size=(32,32))

-get_mean_face(x)

-visualize_mean_face(x_mean, size, new_dims)

Report: Mean face image: ps6-1-a-1.png

1.b. PCA

Now that we have the data points in X and the mean µ, we can go ahead and calculate the eigenvectors and eigenvalues to determine the vectors with largest covariance. See Eigenfaces for Recognition for more details. Using the equation from the lectures:

You need to find the eigenvectors µ. Luckily there is a function in Numpy linalg.eigh that can do this using P as an input.

µT Σµ

where

1 XN T

Σ= (xi −µ)(xi −µ)

M i=1

Code: pca(X, k))

Report: Top 10 eigenfaces: ps6-b-1.png

1.c. Face Recognition (classification)

Now that we have the PCA reduction method ready, let’s continue with a simple (naive) classification method.

First, we need to split the data into a training and a test set. You will perform this operation in the split_dataset function. Next, the training stage is defined as obtaining the eigenvectors from using the training data. Each image face in the training and test set is then projected to the “face space” using these vectors.

Finally, find the eigenface in the training set that is closest to each projected face in the test set. We will use the label that belongs to the training eigenface as our predicted class. Your task here is to do a quick analysis of these results and include them in your report.

Code: split_dataset(X, y, p))

Report: Analyze the accuracy results over multiple iterations. Do these “predictions” perform better than randomly selecting a label between 1 and 15? Are there any changes in accuracy if you try low values of k? How about high values? Does this algorithm improve changing the split percentage p?

2. BOOSTING [20 POINTS]

The Boosting class will contain the methods behind this algorithm. You will use the class WeakClassifier as a classifier (h(x)) which is implemented to predict based on threshold values. Boosting creates a classifier H(x) which is the combination of simple weak classifiers.

Complete the Boosting.train()function with the Adaboost algorithm (modified for this problem set):

Input: Positive(1) and negative (-1) training examples along with their labels; Initialize all the weights to wi ← N1 where N is the number of training examples ; for j = 1…M do

Renormalize the weights so they sum up to 1;

Instantiate the weak classifier h with the training data and labels. Train the classifier h ;

Get predictions h(x) for all training examples x ∈ Xtrain ;

Find ϵj =Pwi for weights where h(xi) ̸= yi;

i

Calculate αj = 12ln(1−ϵjϵj ); if ϵj if greater than a (typically small) threshold then

Update the weights: wi ← wie−yiαj hj (xi); else

Stop the for loop end end

Algorithm 1: Adaboost algorithms

After obtaining a collection of weak classifiers, you can use them to predict a class label implementing the following equation in the Boosting.predict()function:

H(x) = signXαjhj(x)

j

Return an array of predictions the same length as the number of rows in X.

Additionally you will complete the Boosting.evaluate()function, which returns the number of correct and incorrect predictions after the training phase using xtrain and ytrain already stored in the class.

2.a.

Using the Fruits dataset, split the dataset into training (Xtrain) and testing (Xtest) data with their respective labels (ytrain) and (ytest). The naming of the files follows the other datasets so use the load_images() function from the previous section. Perform the following tasks:

1. Establish a baseline to see how your classifier performs. Create predicted labels by selecting N random numbers ∈ {−1,1} where N is the number of training images. Report this method’s accuracy (as a percentage): 100 .

2. Train WeakClassifier using the training data and report its accuracy percentage

3. Train Boosting.train() for num_iterations

4. Train Boosting.train() for num_iterations and report the training accuracy percentage by calling Boosting.evaluate().

5. Do the same for the testing data. Create predicted labels by selecting N random numbers ∈{−1,1} where N is the number of testing images. Report its accuracy percentage.

6. Use the trained WeakClassifier to predict the testing data and report its accuracy.

7. Use the trained BoostingClassifier to predict the testing data and report its accuracy.

Code:

• ps6.Boosting.train()

• ps6.Boosting.predict(X)

• ps6.Boosting.evaluate()

Report:

• Text Answer: Report the average accuracy over 5 iterations. In each iteration, load and split the dataset, instantiate a Boosting object and obtain its accuracy.

• Text Answer: Analyze your results. How do the Random, Weak Classifier, and Boosting perform? Is there any improvement when using Boosting? How do your results change when selecting different values for num_iterations? Does it matter the percentage of data you select for training and testing (explain your answers showing how each accuracy changes).

3. HAAR-LIKE FEATURES [20 POINTS]

3.a.

Youwillstartbygeneratinggrayscaleimagearraysthatcontainthesefeatures. Createthearrays based on the parameters passed when instantiating a HaarFeatureobject. These parameters are:

-type: setsthetypeoffeatureamongtwo_horizontal,two_vertical,three_horizontal,three_vertical, and four_square (see examples below).

-position: represents the top left corner (row, col) of the feature in the image. -size: (height, width) of the area the feature will occupy.

Complete the function HaarFeatures.preview(). You will return an array that represents the Haar features, much like each of the five shown above. These Haar feature arrays should be based on the parameters used to instantiate each Haar feature. Notice that, for visualization purposes, the background must be black (0), the area of addition white (255), and the area of subtraction gray (126). Note that the area occupied by a feature should be evenly split into its componentareas-three-horizontalshouldbesplitinto3evenlysizedareas,four-squareshould be split into 4 evenly sized areas (in other words, divide the width and height evenly using int division).

Code:

• Functions in HaarFeatures.preview()

Report: Using 200×200 arrays.

• Input: type = two_horizontal; position = (25, 30); size = (50, 100) Output: ps6-3-a-1.png

• Input: type = two_vertical; position = (10, 25); size = (50, 150) Output: ps6-3-a-2.png

• Input: type = three_horizontal; position = (50, 50); size = (100, 50) Output: ps6-3-a-3.png

• Input: type = three_vertical; position = (50, 125); size = (100, 50) Output: ps6-3-a-4.png • Input: type = four_square; position = (50, 25); size = (100, 150) Output: ps6-3-a-5.png

3.b.

• ps6.convert_images_to_integral_images

3.c.

Notice that the step above will help us find the score of a Haar feature when it is applied to a certain image. Remember we are interested in the sum of the pixels in each region. Using the procedure explained in the lectures you will compute the sum of the pixels within a rectangle by adding and subtracting rectangular regions. Use the example from the Viola-Jones paper:

Assume the image above is an array ii returned as an integral image. The sum of pixels in D can be obtained by sum(D) = ii(4)−ii(2)−ii(3)+ii(1) where each number corresponds to the pixel position shown in the image. If D was a white rectangle in a Haar feature you will use it as +D, if it was a gray rectangle then it becomes -D.

Important note: Points 1, 2, and 3 in the example above, do not belong to the rectangle D. They cover the areas just before the one in D. Test your approach with the result obtained from D = sum(D)+sum(A)−sum(B)−sum(C).

Complete HaarFeatures.evaluate(ii) obtaining the scores of each available feature type. The base code maps the feature type strings to the following tuples (you will see this in the comments):

• “two_horizontal”: (2, 1)

• “two_vertical”: (1, 2)

• “three_horizontal”: (3, 1)

• “three_vertical”: (1, 3)

• “four_square”: (2, 2)

Code:

• HaarFeatures.evaluate(ii)

Report: Text Answer: How does working with integral images help with computation time?

Give some examples comparing this method and np.sum.

4. VIOLA-JONES [30 POINTS]

Haar-like features can be used to train classifiers restricted to using a single feature. The results from this process can be then applied in the boosting algorithm explained in the Viola-Jones paper. We will use a dataset of images that contain faces (pos/ directory) and refer to them as positive examples. Additionally, use the dataset of images in the neg/ directory which contain images of other objects and refer to them as negative examples. Code the boosting algorithm in the ViolaJones class.

First, we will be using images resized to 24×24 pixels. This set of images are converted to integral images using the function you coded above. Instantiate a ViolaJones object using the positive images, negative images, and the integral images. ViolaJones(train_pos, train_neg, integral_images). NoticethattheclasscontainsoneattributetostoreHaarFeatures(self.haarFeatures) and another one for classifiers (self.classifiers).

WehaveprovidedafunctioncreateHaarFeaturesthatgeneratesalargeamountoffeatureswithin a 24×24 window. This is why it is important to use integral images in this process.

Use the Boosting algorithm in the Viola-Jones paper (labeled Table 1) as a reference. You can find a summary of it below adapted to this problem set.

For each integral image, evaluate each avaliable Haar feature and store these values. Initialize the positive and negative weights: wpos = 21p and wneg = 21n ; for t = 1…T T: number of classifiers (hypotheses) do

1. Normalize the weights,

wt,i

wt,i ← n

P wt,j j=1

2. Instantiate a and train a classifier hj using the VJ_Classifier class ; For each feature, j, obtain the error evaluated with respect to wt ; ϵj =Pi wi for cases where hj(xi) ̸= yi ;

ϵj can be obtained from the error attribute in VJ_Classifier ;

Train hj using the class function train(). This will fix the classifier that returns lowest error ϵt. ;

3. Append hj to the self.classifiers attribute ; 4. Update the weights: ;

wt,i ← wt,iβ1t−ei

ϵt βt = −

1 ϵt

if ht(xi) == yi then ei = negative

else ei = positive

end

5. Calculate:

1

αt = log( )

βt

Append it to self.alphas end

Algorithm 2: Adaboost algorithms Using the best classifier ht, a strong classifier is defined as:

T

Positive label if P t=1

H(x) =

T P αt else Negative label

t=1

4.a.

Complete ViolaJones.train(num_classifiers)with the algorithm shown above. We have provided an initial step that generates a 2D array with all the feature scores per image. Use this when you instantiate a VJ_Classifier along with the labels and weights. After calling the weak classifier’s train function you can obtain its lowest error ϵ by using the ‘error’ attribute.

Code: ViolaJones.train(num_classifiers)

4.b.

Complete ViolaJones.predict(images)implementing the strong classifier H(x) definition. Code: ViolaJones.predict(images)

Report:

-Output: ps6-4-b-1.png and ps6-4-b-2.png which correspond to the first two Haar features selected during the training process.

-Text Answer: Report the classifier accuracy both the training and test sets with a number of classifiers set to 5. What do the selected Haar features mean? How do they contribute in identifying faces in an image?

4.c.

Now that the ViolaJones class is complete, you will create a function that is able to identify faces in a given image. In case you haven’t noticed we’re using images that are intended to solve the problem below. The negative directory contains patches of a scene where there is people in it except for their target face(s). We are using this data to bias the classifier to find a face on a specific scene in order to reduce computation time (you will need a larger amount of positive and negative examples for a more robust classifier).

Use a 24×24 sliding window and check if it is identified as a face. If this is the case, draw a 24 x 24 rectangle to highlight positive match. You should be able to only find the man’s face. To that extent you will need to define a positive and negative datasets for the face detection. You can choose images from the pos/ and neg/ datasets or build your own from the man.jpeg image by selecting subimages.

Code: ViolaJones.faceDetection(image, filename=None) Report:

Use the following input images and return a copy with the highlighted face regions.

-Input: man.jpeg. Output:ps4-4-c-1.png

5. EXTRA CREDIT: CASCADE CLASSIFIER [10 POINTS]

In this last section, we will use the Haar-features and Viola Jones classifier to build a cascade of classifiers for face detection. This section is extra credit.

5.a.

First, we will write the functions predict and evaluates_classifiers to evaluate a set of cascaded classifiers. We represent the set of cascaded classifiers as a list where each index is a Viola Jones classifier object. Within the cascade, only positive results from the i’th classifier get passed to the i+1 classifier. At any stage, if a classifier returns negative, then the entire cascade returns negative.

Thepredictfunctiontakesanimageandreturnspositiveornegativebasedonifafaceispresent. The evaluates_classifiers takes a set of positive and negatives images and reports the detection rate, false positive rate, and a list of the false positive images. Code:

• CascadeClassifier.predict(classifiers, img)

• CascadeClassifier.evaluate_classifiers(pos, neg, classifiers)

5.b.

Now we will write the training routine for the cascaded classifier. Use the Cascade algorithm in the Viola-Jones paper (labeled as table 2 in the paper) to build the cascaded classifier. We’ve provided an adapted version for this problem set below. Also complete the face detection function similar to ViolaJones.

f_target, f_max_rate, and d_min_rate set before ; train_pos = list of positive training images ; train_neg = list of negative training images ; validate_pos = list of positive validation images ; validate_neg = list of negative validation images ; f = f_last = 1.0 ; d = d_last = 1.0 ; i = 0;

while f &gt; self.f_target do initialize vj as ViolaJones classifier with train_pos and train_neg ;

create integral images with train_pos and train_neg ; while f &gt; f_max_rate * f_last: do ni += 1 ; vj = Train a VJ classifier with train_pos, train_neg, and integral images and ni classifiers ;

Evaluate cascaded classifier with new vj to calculate new f and d ; while d &lt; self.d_min_rate * d_last do

Decrease threshold for vj;

Re-evaluate cascaded classifier with new threshold to calculate f and d; end

f_last = f ; d_last = d ;

Add vj to cascaded classifiers list ;

N = null set ;

Evaluate current cascaded detector and put any false detections into N ; end

Algorithm 3: Cascade Classifier algorithm Code:

• CascadeClassifier.train(classifiers, img)

• CascadeClassifier.faceDetection(image)

Report:

-Text Answer: Report the cascaded classifier accuracy on both the training and test sets. What was the best percentage for the train/test split? What values did you choose for the false positive target, the false positive rate, and the detection rate? What impact did these have on the overall cascaded classifier?

-Text Answer: How many classifiers did your cascade algorithm produce? How many features did each of these classifiers have? Compare this classifier to just a single Viola Jones classifiers.

-Image Answer: Include an image you selected and the faces detected on the image. Choose any image of your liking except for the image given in section 4 (man.jpeg).
