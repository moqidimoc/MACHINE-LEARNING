# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

# ********************************************* (the code between these marks is the one that I have added)
# INTRODUCTION TO THE PROBLEM

# The coursework description asks us to construct a classifier to manage PAC-MAN. This classifier has to
# choose the best option to achieve its goal, eat all the food of the map without being eaten by the ghosts.
# To do so, our classifier will receive a feature vector of 25 binary features. These features encode 
# different aspects of the game, like for example: are they ghosts around? is there food around? or is there
# any wall in front? I do not actually know which feature corresponds to each one of these questions, but as
#  it is not the scope of the coursework, I will not investigate further.

# We will have to code a Multicass classifier, it has to decide the best/more logical option between four
# different classes (0, 1, 2, 3). These classes actually encode  four possible directions in the map (NORTH,
# EAST, SOUTH, WEST), see the function convertNumberToMove. These classes will decide in which direction 
# PAC-MAN will move having in mind the different 25 features we talked about before.

# In general, every Machine Learning model has four distinct steps. The first one is usually Preprocessing.
# In this step we arrange and clean our data: fill missing values or drop them, encode our categorical features
# so they are better ingested by our model, remove superfluous features or split our data in train and test
# (which we will not do). As this procedure is already managed in the function convertToArray, we can omit
# this step and focus our time in the next three steps: Learning, Evaluation and Prediction. The most important
# one today, and the one we will pay the most attention to, will be the Learning of our model. As we do not
# have so many data rows (126 data instances), the training is going to be small. Ergo, the model is not going
# to be a state-of-the-art one, as it is normal. However, this is not the goal of the coursework so we will not
# cudgel our brains seeking the best performance. The Prediction will be done in getAction(). Here we will tell
# PAC-MAN what to do depending on the feature vector it recieves. It will make a decision based on the weights
# it has learned from the training. Regarding the Evaluation, we will leave that to the professors marking this
# script.

# The model we are going to use is a simple Multiclass Perceptron. There are several approaches for a multiclass
# problem. However, we will use this one because it works well with small data and it is easy to implement. As a
# side note, I will leave a function I named RBF() that implements a Radial Basis Function (RBF) Neural Network.
# I have tried to use this model as the classifier. However, it has struggled a lot with the data it recieves.
# I believe it is because there are data instances repeated with different targets (some times you choose one move,
# and other times you choose another one, even though the conditions (feature vector) are the same). Another issue
# I think this model has is the features it recieves, there are 0 or 1, and that affects the performance in some
# manner I cannot interpret. Anyway, as the Perceptron is the closest to an ANN, I will use this as my classifier. 

# All the same, the main problem is the amount of data we have to train our model. So, the model we choose is not
# going to be that differential. The quality leap will come from the training of that model. And for that we need
# as much data as we can get. Therefore, we will try to add information to our model with some custom training data
# we can generate by playing PAC-MAN with TraceAgent. This data is stored in another file called 'moves.txt'. By
# adding data to our training we may obtain better results than with a more powerful model used only with
# 'good-moves.txt'. In registerInitialState(), where the training occurs, we have set a variable called filename.
# If there is more training data available, you can introduce in that variable the name of the file where the 
# additional data is. So it is also used in addition to the training data in 'good-moves.txt'.
# *********************************************

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.

    # *********************************************
    # Classifier. We will implement the Multiclass Perceptron as a function so we can call it as much times as 
    # we need. I have based this model in some slides I saw in another module in this MSc (Pattern
    # Recognition, Neural Networks, and Deep Learning).

    # Main idea: our classifier will assign the feature vector Yk to the class Wj if Gj(Y)>Gi(Y) for every
    # i different from j. If classification is wrong, we adjust weights of the discriminant functions:
        # move weights for required class TOWARDS input,
        # or, move weights of wrongly selected class AWAY from input.

    def multiclassPerceptron(self, data, target, learning_rate=0.01, initial_weights = []): # we leave some values predefined in case they are not passed to the function
    																						# in initial_weights you could pass some pretrained weights
    																						# in learning_rate you could pass a different learning rate if desired
    	y = np.array(data) # Here we store the data we have
    	w = np.array(target) # Here we store the targets of our data

    	# The following variables will be accesible from outside of this function
        self.n_classes = max(w)+1 # This is the number of possible classes, assuming one possible class is 0
        self.n_instances = len(y) # This is the number of instances our dataset has
        if len(y[0])>0:
        	self.n_features = len(y[0]) # This is the number of features each datarow has, assuming it has at least one entry
        else:
        	self.n_features = 0

        # Here are the different steps we have to follow to train our model:

        # 1. Set a value of hyper-parameter eta.
        eta = learning_rate


        # 2. For each possible class, c (c=4), initiliase a_c (our augmented vector) to an arbitrary solution.
        # We will put '_c' as a suffix to distinguish one class from another. Technically, a_1 is the same as a[0],
        # a_2 is the same as a[1] and so on. When we talk about the augmented vector, we mean that we extend a vector,
        # or matrix, by appending elements to it. So, instead of defining G(x)=wT*Y+w_0, we define: G(x)=aT*Y,
        # where ac=[w_0, w_cT]T and y=[1, xT]T (T means transposed).

        # As we have 4 possible classes (0, 1, 2, 3), we need to set a by defining w_0 and w_c, where c={0, 1, 2, 3}.
        # To do all this in one line, we will help ourselves with the random library, so we will create a random
        # array with 4 rows and 26 columns (one row per class, and one column per feature and another one for the bias, w_0).

        # We will initialise all of them to a random value more or less between -1 and 1. We will do so by calling
        # np.random.randn that creates a M(number of classes)xN(number of features+1) matrix with float values
        # randomly sampled from a univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

        # ATTENTION: if we have pretrained our model, we already should have some initial weights, so we should
        # only do this if there are no initial weights. We can do so if the initial_weights variable is empty, as we
        # setted this variable to a default value: an empty list, it is easy to know if it is already pretrained. I
        # added this part in case there would be some pretraining involved, another way of dealing with small data.
        if len(initial_weights) == 0:
            a = np.random.randn(self.n_classes,self.n_features+1) # One row per class, and one column per feature and an extra one for the bias (w_0)
            y = np.insert(y, 0, 1, axis=1) # we add 1 in y into the position 0 (at the beginning), so it multiplies the bias (data augmentation)

        else: 
            a = initial_weights # If we have already passed some pretrained weights, use those weights to start
            y = np.insert(y, 0, 1, axis=1) # we add 1 in y into the position 0 (at the beginning), so it multiplies the bias (data augmentation)


        # 3. For each sample, (Yk, wk) in the dataset in turn (Yk is the k-ith element of our data set, and wk is
        # the class/target of that k-ith element)

            # 3.1 Classify: c'=argmax(G_c(Yk))
        # We can either choose our learning to terminate after a fixed number of epochs, or we can choose the learning to stop when our weights
        # stop changing. As our dataset is rather small, we will not need a long training, so we can set the number of epochs to 200 to be sure
        # the learning is done. Usually, a reasonable number of epochs is between 100 and 150.
        n_epoch = 200 # We set a fixed number of epochs

        for epoch in range(n_epoch): # we repeat this process a fixed number of times
            for n in range(self.n_instances): # We will do it sequentially, update the weights for each instance (if necessary)
                result = np.zeros(self.n_classes) # we initialise the array where we will store the outputs of each G_c

                for i in range(self.n_classes): # we have to calculate four different functions, one per each class (G_c is the same as result[c-1])
                    result[i] = np.dot(a[i], y[n]) # we estimate the value of each G_c and store it in result
                final_result = np.argmax(result) # This variable will contain the output of our classifier, it will be the largest output of G_c

            # 3.2 Learning: Now we have two possible options, the classifier worked correctly, and the final_result is the same as the target(w),
            # nothing has to be done. On the other hand, if our classifier misclassified that specific feature vector, we have to move two weight vectors:
                # Let's imagine that an instance y[n] was classified with final_result=0 and it actually was w[n]=1. In this case we
                # have to (1) move the weights vector a[1] towards y[n] and (2) move the weights vector a[0] away from y[n].

                if final_result != w[n]:
                    # (1): move a[w[n]] towards y[n]
                    a[w[n]] += eta*y[n]
                    # (2): move a[final_result] away from y[n]
                    a[final_result] -= eta*y[n]

        # Once the training is done, after 200 epochs, we already have our trained weights. This will be the only thing that our function will return.
        return a
    # *********************************************


    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        # This part of the code is where the training happens. After initialising the program, our model (implemented before in multiclassPerceptron) 
        # will learn from the data it has at hand. The more data it has, the better it will perform, so we will try to gather as much data as possible.

        # First, I will TRY to read some custom training data so our classifier is better trained. In case an EXCEPTION is arised, this part of the code
        # will be omitted. We will open the 'moves.txt' datafile IF it exists, so if there is not any custom training data, this part of the code will
        # not affect, and our classifier will not crash. In case anyone wants to feed the classifier with other file different from 'moves.txt', he/she
        # should just change the content in the variable filename and store the name of the file.

        # I found this piece of code in https://stackoverflow.com/questions/15032108/pythons-open-throws-different-errors-for-file-not-found-how-to-handle-b/15032444
        # It manages errors both from MacOS and from Windows, so it should work good with both systems.
        filename = 'moves.txt' # You can change here the name of the file you want to open to aggregate training data

        try: # the try clause will open the file if it exists, if except occurs, it will not do nothing
            self.file = open(filename, "r")
            train_content = self.file.readlines()
            self.file.close()

        except (OSError, IOError) as e:
            train_content = 'empty'

        if train_content != 'empty': # If this file actually exists, this value will not be 'empty', so we convert this training data to an array.
        # We will reuse the code from above, but now storing it in train_data and train_target
            train_data = []
            train_target = []
            # Turn train_content into nested lists
            for i in range(len(train_content)):
                lineAsArray = self.convertToArray(train_content[i])
                dataline = []
                for j in range(len(lineAsArray) - 1):
                    dataline.append(lineAsArray[j])

                train_data.append(dataline)
                targetIndex = len(lineAsArray) - 1
                train_target.append(lineAsArray[targetIndex])

        # If this file actually existed, we have more training data at hand. In order to exploit this additional data, we will add it to the data we  
        # had in 'good-moves.txt' so our model has more information from which it could learn. 

        if train_content != 'empty': # if it exists we concatenate both arrays
        	self.final_data = np.concatenate((np.array(self.data), np.array(train_data)))
        	self.final_target = np.concatenate((np.array(self.target), np.array(train_target)))

        else: # if not, we only use the data from 'good-moves.txt'
        	self.final_data = np.array(self.data)
        	self.final_target = np.array(self.target)

        # Now we will train our model with the information at hand. It could be that this information is only made up with the moves in 'good-moves.txt', or 
        # it could be that this information is made up with the one in 'good-moves.txt' enriched with the one in the filename (in our case, 'moves-txt')
        self.weights = self.multiclassPerceptron(self.final_data, self.final_target)
        # *********************************************



    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"

        # *********************************************
        # If the classifier winned, we congratulate the Perceptron for its amazing job (I sincerely doubt it will happen, but if it does he deserves
        # all the respect possible)
        if state.isWin():
            print "Congrats Perceptron! You winned! Who would have expected that!!"

        # If the classifer losed, which will most surely happen the 99.99% of the times, we encourage him to keep trying
        else:
            print "Keep trying Perceptron, you're getting there!"
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        # In this part of the code the predictions are made. The last line of code collects the features that represent the state of the game in a 
        # given time. With these features, PAC-MAN has to make a decision to where to move. He will make that decision helped by the model trained
        # in registerInitialState. Now that the model is trained, the weights of our perceptron are stored in self.weights. 

        # Here I insert code to call the classifier and decide where to go based on the features.

        # First, we add 1 in features into the position 0 (at the beginning), so it multiplies the bias (data augmentation)
        augmented_features = np.insert(features, 0, 1) 

        # Just like in the training, we have to choose the function with the largest output, so we have to calculate four different outputs, one
        # for each class.
        results = np.zeros(self.n_classes) # we initialise the array where we will store the outputs of G_c(x)
        for i in range(self.n_classes): # one dot product per class
            results[i] = np.dot(self.weights[i], augmented_features) # in results we store the value of each output
        prediction = np.argmax(results) # and we choose the most logical move for our classifier

        # Here we use the function from before to convert the prediction in a movement
        direction = self.convertNumberToMove(prediction) 
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)

        # *********************************************
        # We remove STOP if it is one of the possible options so our PAC-MAN does not stop come hell or high water
        if Directions.STOP in legal: 
            legal.remove(Directions.STOP)

        # Now that we have the direction we would like to follow stored in the variable direction, and that we also have the possible directions 
        # we can actually follow stored in legal. Our approach will be the following:

        # 1. If we can move in the direction predicted, we move that way
        if direction in legal:
            return api.makeMove(direction, legal)

        # 2. In case we could not move in the predicted direction (1.), we choose a random choice from the options remaining (without stopping).
        # I have put this just in case, because the algorithm learned from a game where these 'illegal' actions could not happen.
        else:
            pick = random.choice(legal)
            return api.makeMove(pick, legal)

        # NOTE: we could have mimicked the logic in RandomishAgent, but I think that sometimes it is better to change your last move. If not,
        # PAC-MAN would not do some moves that could save him in some specific contexts. Additionally, in RandomishAgent PAC-MAN only moves in 
        # a rectangle of the map because it cannot enter through specific paths as it is impossible to come from that direction.
        # *********************************************

# *********************************************
# FINAL CONCLUSION TO THIS COURSEWORK

# Obviously, my model has a lot of flaws. Some of these flaws could have been avoided with more training data. I believe this is the main issue
# for this problem, and for many others. At the end, Machine Learning is "a set of methods that can automatically detect patterns in DATA, and
# then use the [...] patterns to predict future data, or to perform other kinds of decision making [...]", as Murphy said. For me, the main word
# in this statement is DATA. If you have 2 million data points and perform a 1-nearest neighbour classifier, it will surely perform better than 
# this Multiclass Perceptron with 126 instances, or even better than a super Neural Network. It will primarily be because of the training data size.

# I have ascertained this point made above by testing my classifier with different conditions:

	# 1. When I run ClassifierAgent without a 'moves.txt' file created (without playing with TraceAgent), the classifier works good but gets stuck 
	# easily. It does not get a good score at all.

	# 2. When I run ClassifierAgent after generating a small 'moves.txt' (winning faster). I may have a better score, but I have less data instances. 
	# Now PAC-MAN plays better, but keeps getting stuck in some simple moves. 

	# 3. When I run ClassifierAgent after generating a big 'moves.txt' (winning in a longer time). I have a worst score but, instead, I have much more
	# data instances. My PAC-MAN has learned from movements where there was not much food, or no food at all around. So now he is capable of taking a 
	# decision even though there is not any food around (he did not move like this in condition 1.). 

# The conclusion I draw from the three different conditions from above is: the more data my model ingests for learning, the better it predicts. Even
# though the score is lower, or even though it loses. In fact, if PAC-MAN wins or loses with TraceAgent does not matter. What matters is that in ingests
# as much information as possible. 

# There are some conditions in which PAC-MAN works awfully bad. The principal condition is when it reaches a corner without food around. It starts going
# back and forth repetitively. I suppose that with more instances where PAC-MAN get's out of the corner without food around would help him to take a 
# better decision.

# To be honest, I loved this coursework. It has improved my coding greatly. I would have loved that my RBF NN would have worked, but even though it did 
# not, it has made me raise questions from other angles of sight. 
# *********************************************

# *********************************************
# BIBLIOGRAPHY: articles or webpages I used to elaborate this script

# stackoverflow.com -> to resolve main programming doubts or to find better approaches to some problems
# numpy.org -> NumPy documentation
# docs.scipy.org -> NumPy documentation
# https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab (1)
# https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/ (2)
# (1) and (2) where articles I read to approach the limited data situation. It did not be of such assistance as I thought because
# it was focused specially on small IMAGE datasets. Either way, it was really interesting and instructive. I was not fully aware of 
# the concept of TransferLearning or Pretraining, and it is most useful.
# https://www.researchgate.net/post/How_does_one_choose_optimal_number_of_epochs -> to find out a good number of epochs
# https://keats.kcl.ac.uk/pluginfile.php/6768496/mod_resource/content/2/Lecture6.pdf -> information for the RBF NN. I used the slides 40-52.
# I tried to explain as good as possible this in the comments of the code. However, these slides are much more illustrative. 
# https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-for-classification-problem-33c467803319
# Medium article which explains this approach too.
# *********************************************


# *********************************************
# SIDE NOTE: Radial Basis Function (RBF) Neural Network

	# Classifier. We will implement the RBF (Radial Basis Function) NN as a function so we can call it as much times as 
	# we need. I have based this model in some slides we saw in another module in this MSc (Pattern Recognition, Neural
	# Networks, and Deep Learning).

	# Main idea: a RBF Neural Network is a three-layer network: input, hidden and output layers.
	    # Input unit: linear transfer function 
	    # Hidden unit: radial basis function (RBF), h(). This is the main difference with the typical Feedforward NN.
	    # Output unit: any activation function, f(). In our case, we will use a linear output function to simplify 
	                                                # the second part of our training (Supervised Learning part,
	                                                # determining the output weights W).
	            
	# The training phase has different parts:

	    # 1. Determine c, sigma and the activation functions (Unsupervised Learning part).
	        # c will be the centre of our RBF in the hidden units, these centres can be preset or determined by a
	        # training algorithm. In our case we will determine these centres by selecting some samples selected at
	        # random. sigma is also needed for the Gaussian function we will use as a basis activation function (RBF).
	        
	    # 2. Compute Y(x) using the training examples.
	        # In this step, once we have c, sigma and the activation function (Gaussian function), we will calculate
	        # the outputs of our hidden layer by simply introducing the training samples to the hidden layer. This
	        # will create a matrix, which we will call phi.
	        
	    # 3. Determine the weights at the output layer using Y(x)/phi by least square methods. (Supervised Learning part).
	        # By helping ourselves with linear algebra (NumPy) we will calculate the output weights. This will result 
	        # in an array of size n+1 (n will be the number of nodes/centres in the hidden layers, the +1 is the bias).

	def RBF(self, data, target, initial_weights = []): # we leave some values predefined in case they are not passed to the function
	    x = np.array(data) # Here we store the data we have
	    t = np.array(target) # Here we store the targets of our data

	    # The following variables will be accesible from outside of this function
	    self.n_classes = max(t)+1 # This is the number of possible classes, assuming one possible class is 0
	    self.n_instances = len(x) # This is the number of instances our dataset has
	    if len(y[0])>0:
	    	self.n_features = len(y[0]) # This is the number of features each datarow has, assuming it has at least one entry
	    else:
	    	self.n_features = 0

	    # Here are the different steps we have to follow to train our model:

	    # 1. Determine c, sigma and the activation functions. As we said:
	        # Determination of centres: fixed centres selected at random. This has some advantages, it is simple and quick,
	        # no learning involved. The number of hidden units (nh) is less than the number of input patterns (nh<n_instances).
	        
	        # Procedure:
	            # 1.1: choose number of hidden nodes (nh).
	    nh = int(self.n_instances/20)+1 # we make a hidden node for each 20 instances and we round upwards
	    
	            # 1.2: pick randomly nh input patterns from the data set (with n input patterns) as the centres (c).
	    c = np.zeros((nh, self.n_features))
	    rng = np.random.default_rng() # to have non-repetitive random numbers
	    for i in range(nh):
	        c[i] = x[rng.choice(self.n_instances, replace=False)] # with replace=False, we will not have two centers with the same instance

	    # NOTE: I believed that we could not have two centres with the same feature array, but I was mistaken. There are a lot of repeated
	    # instances in 'good-moves.txt', as it is normal. There a lot of moves that have a similar context (same feature vector). Furthermore,
	    # there a lot of repeated features with distinct targets. I found about this investigating throught the dataset. From the 126 instances 
	    # 'good-moves.txt' there are like 35 unique feature instances. This could be one of the reasons of why my model did not perform correctly.

	            # 1.3: when gaussian function is employed, define sigma=rho_max/pow(2*nh, 0.5) or sigma=2*rho_avg for all centres.
	            # rho_max is the maximum distance between the chosen centres and rho_avg is the average distance between the 
	            # chosen centres. We will choose rho_avg to ease things up. As we have nh centers, we should calculate the 
	            # distance between all of them and then divide them by the number of distances to have the rho_avg.
	    distances = np.array([])
	    for i in range(nh): # for every centre
	        for j in range(i+1, nh): # calculate the distance with the other centres
	            distances = np.append(distances, np.linalg.norm(c[i]-c[j])) # np.linalg.norm(a-b) is the euclidean distance between a and b
	    rho_avg = distances.mean() 
	    sigma = 2*rho_avg # sigma=2*rho_avg. This value will be the same for all our centres
	    
	    # 2. Compute Y(x) using the training examples. We will do so helping ourselves with NumPy. We have to calculate
	    # all the outputs from the hidden layer, this is Y(x). We will store these values in a 2D array (nmatrix) which
	    # we will call phi
	    phi = np.zeros((self.n_instances, nh+1)) # we initialise the matrix where we will store the outputs. 
	                                        # one row per instance and one column per hidden layer plus the bias
	    for i in range(nh):
	        phi[:, i] = np.array(gaussian(x, c[i], sigma)) # here we insert y(c[i]) that is the output of the node i with the gaussian function defined before
	    phi[:, nh] = [1]*self.n_instances # Lastly, we add the bias, what the last node of the hidden layer would output: 1
	    
	    # 3. Determine the weights at the output layer using Y(x)/phi by least square methods. This step depends on the form
	    # of our phi matrix. If it is squared, we can calculate the ouput weights (W) by multiplying the inverse of phi by
	    # the targets. However, we will assume it is a non-square matrix, so our weights will be calculated as it follows:
	        # We know that: phi(x) * W(weights) = T(x)(targets)
	        # so, by applying some algebra we have:
	            # phi.T * phi * W = phi.T * T
	            # Now we should pass the first element of the left side (phi.T * phi) to the other side.
	            # And finally: W = inverse(phi.T*phi) * phi.T * T
	    weights = np.dot(np.dot(np.linalg.pinv(np.dot(phi.T,phi)), phi.T), t) # in this variable we will store the weights
	    
	    # NOTE: we used np.linalg.pinv instead of np.linalg.inv because if the determinant of the matrix (phi.T*phi) is zero, it
	    # will not have an inverse, thus the second function will not work. On the other hand, np.linalg.pinv will work even
	    # though the determinant is zero. This is because this function returns the inverse of your matrix when it is available,
	    # and the pseudo-inverse when the determinant is zero. The different results of the functions are because of rounding
	    # errors in floating point arithmetic.
	    # I found about this in Stack Overflow: https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers
	    
	    # weights should be an array of size=nh+1. Each value in this array corresponds to the output weight of each hidden
	    # node. So in weights[0] we have W11 (output weight from the first hidden node to Z1, unique output node), in 
	    # weights[nh-1] we would have W1nh (last output from our hidden layer), and in weights[nh] we would have W10 (the
	    # output weight of the bias)
	    

	    # NOW, OUR MODEL HAS ALREADY LEARNED, THE TRAINING IS OVER AND WE HAVE OUR RBF NEURAL NETWORK IMPLEMENTED

	    return c, sigma, weights # all that we need to calculate the output of our RBF NN

	# We already have the NN implemented. Now we know that the output of our NN will be the following:

	    # z(x) = W11*gaussian(x, c1, sigma) + W12*gaussian(x, c2, sigma) + ... + W1nh*gaussian(x, cnh, sigma) + W10
	    
	# Where: 
	    # z(x) is the ouput we are searching for, the direction PAC-MAN should follow
	    # Wkj are the output weights from the node j of the hidden layer to the node k of the output layer (we only have 1)
	    # gaussian(x, cn, sigma) is the output of the hidden node n, which will be multplied by the output weight
	    # W10 is the bias
# *********************************************