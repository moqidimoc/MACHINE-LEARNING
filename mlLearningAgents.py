# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

# ********************************************* (the code between these marks is the one that I have added)

# INTRODUCTION TO THE PROBLEM

# The coursework description asks us to construct a Reinforcement Learning model to manage PAC-MAN. This model has to make
# PAC-MAN choose the best option to achieve its goal, eat all the food of the map without being eaten by the ghosts. To do so,
# our model will receive a state object and choose the best action; this will be done in the getAction() function, where the
# program chooses which action to do at each step. This state object encodes different aspects of the game, like for example:
# where are the ghosts? where is the food around? 

# We will have to code a Reinforcement Learning model, it has to decide by its own the best/more logical option between four
# different actions. These actions actually encode four possible directions in the map (NORTH, EAST, SOUTH, WEST, after removing
# STOP). Each time PAC-MAN has a move to make, or choose an action, the possibilities are obtained through the function
# getLegalPacmanActions(), and stored in the variable 'legal'. 

# Hence, the main goal of this Coursework is to build a Reinforcement Learning model. As an introduction, we will briefly talk
# about this type of models. They do not have much to with the other models we have seen so far (Supervised and Unsupervised 
# Learning). RL is learning what to do, how to map situations to actions, to maximise a numerical reward signal. This type of 
# ML technique enables an agent to learn in an interactive environment by trial and error using feedback from its own actions
# and experiences. The environment could be any world in a videogame, like Dunkirk in a World War II game, or like the corresponding
# football pitch in FIFA 21; in our case, the environment is the world where PAC-MAN is, it could be the SmallGrid or any other.
# In this environment PAC-MAN encounters ghosts and food. The agent will be the one and only PAC-MAN, and he will learn by trial
# and error. Therefore. the agent will learn from all the games it will play. The games it will learn from will be the episodes
# set for learning, then it will apply what it has learned to the test games, where we will test how well it does. This last part
# corresponds the the last five lines of code. For a certain number of episodes, alpha (learning rate) and epsilon (exploration
# rate) are set to 0. Thus, no more training nor exploration is done, for these last episodes, the agent plays by its own with
# the prior knowledge obtained. It only used the q_table updated during the previous training episodes. If we set epsilon to zero, 
# the learner will always do what it thinks is best, and it will not get killed for doing something random. Similarly, if we set 
# alpha to zero, the update does not change the Q-values table.

# The model we are going to use is a Q-Learning model. This is a model-free approach to ACTIVE Reinforcement Learning. Active 
# Reinforcement Learning agents must DECIDE what to do (while learning). The main difference with Passive RL is that the agent
# is TOLD what to do, he is given a policy (method to map agent's state to actions) to follow. This kind of model revolves around 
# the notion of a Q-value of a state/action pair, Q(s, a); where 's' is the state, and 'a' is the action. So, basically, when we
# are in a state we look at the Q-values of all the possible actions (a) for that state, and select the best one: U(s)=max[a](Q(s, a)).  
# This part of the learning is known as exploitation (the agent exploits known information to maximise reward, leading to an inmediate
# reward). In this part, we want to achieve the best action, we maximise in 'a'. Therefore, we have the following update rule for 
# the Q-vale of a (state, action):

#                       Q(s, a) <- Q(s, a) + alpha * [ R(s) +  gamma * max[a']( Q(s', a') ) - Q(s, a) ]

# We recalculate it every time that a is executed in s and takes the agent to s', a' are all the actions we know about in s' (the next
# state). Since Q-learning is an Active approach to Reinforcement Learning, we have to choose which a' to select in s'. We could use 
# epsilon-greedy or force exploration. In our model, we are going to use epsilon-greedy. This second part of the model is known as
# exploration (finds more information about the environment and might enable higher future reward). Exploration is really important in
# Reinforcement Learning, it helps to find the optimal action earlier.

# The previous update rule could be translated to something like:

#                       New_Estimate <- Old_Estimate + Step_Size * [Target - Old_Estimate]

# Where:
        # New_Estimate: Q(s, a) at the left side of the equation, the new Q-value for this (state, action)
        # Old_Estimate: Q(s, a) at the right side of the equation, the Q-value we had before for this (state, action).
        # Step_Size: alpha, or the learning rate.
        # Target: R(s) +  gamma * max[a']( Q(s', a') ). This is the reward of taking an action, it can be positive (if the previous action
        # 		  beneficial) or negative. Gamma is the discount rate, it represents the preference of an agent for current reward over
        #		  future reward. Gamma multiplies the maximal Q-value for the next state between a range of options.
        # [Target - Old_Estimate]: is an error in the estimate which is reduced by taking a step towards Target.

# It is important to notice that we run an update after having moved from s to s'. In short, we need:
        # A list of state/action pairs <s_i, a_j> to create the q_table that will store all the possible Q-values.
        # Each state/action pair has Q(s_i, a_j).
        # For a given s_i, just pick the a_j to maximise Q(s_i, a_j).


# Note: I have commented the parts of the code where it prints the legal moves, PAC-MAN's position, the ghost position, the food location,
# the score and when the game ends. This can be uncommented to check in the command line how the training is going.

# *********************************************


from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0


# *********************************************

		# This will correspond to the first piece of the coursework: something that learns, adjusting utilities based on how well
		# the learner plays the game.

        # We will follow the pseudocode from the module slides (slide 48 of Reinforcement Learning 2).

        # In the constructor, init() function, we will initialise the persistents too. These are variables that will change at each
        # step during each training game. They will be set as global variables to access them whenever we need.

        # First, we will initialise the Q-value table. This is a table of Q-values indexed by state and action. Later on, we will
        # update this table as PAC-MAN does a certain action in a certain state. In the 'util.py' script a counter is already
        # defined, and we will benefit from this predefined counter. It works very similarly to a Python dictionary. All its values 
        # are initialised to zero, regardless of the key.
        self.q_table = util.Counter() # This creates an object like this: q_table = {}.

        # In the pseudocode of the module slides appears a table of frequencies for state-action pairs. As we will use the
        # epsilon-greedy approach instead of the frequency-based policy, this table is not needed.

        # We also have to initialise the previous state, action and reward. These values are initially null as the PAC-MAN has not
        # started to play yet. The constructor only initialises these objects.
        self.s = None # Last state.
        self.a = None # Last action.
        self.r = 0 # Last score, it will help us to calculate the reward.


    # Now we will define a function that be in charge of the updates of the Q-values of the q_table. It will work when we are in a 
    # particular state. We will call this function later on, in getAction() (function called by the game every time that PAC-MAN wants
    # to make a move, which is at every step of the game).

    # This function is the one in charge of doing the update.
    def update(self, state, action, reward, q_max):

    # 1. We define the elements we will need for the update. From the INTRODUCTION, we will need an Old_Estimate, a Step_Size, and
    # a Target, which contains the parameters 'reward' and 'q_max' defined at the beginning.
        old_estimate = self.q_table[(state, action)] # Q(s, a) at the right side of the equation, last estimate of the Q-value.
        step_size = self.getAlpha() # alpha, or the learning rate.
        target = reward + self.getGamma()*q_max # R(s) +  gamma * max[a']( Q(s', a') ).

    # 2. We run the update for the New_Estimate, the updated value for the Q-value of this particular (state, action) pair:
        self.q_table[(state, action)] = old_estimate + step_size*(target - old_estimate)


    # For the update, we will need a function to return the maximal Q-value of a state. For a given state we have to see which action
    # would return the highest Q-value.
    def q_max(self, state):

    	# Here we will store the possible Q-values to later return its maximum. 
        q_values = []

        # Here we get the possible actions for a given state, for example: ['West', 'North']. Then, we obtain each Q-value from the
        # q_table where the Q-values are stored. So, in the previous example, we would have a Q-value for the given state and for the 
        # action 'West', and another for the given state for and the action 'North'.
        legal = state.getLegalPacmanActions()

        # First, we remove the 'STOP' direction.
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Nest, we obtain the Q-values for each remaining action:
        for action in legal:
            q_value = self.q_table[(state, action)] # We obtain the Q-value for the given action.
            q_values.append(q_value) # We append it to the list with all the Q-values for each possible action.

    	# Now we return the maximum. From the previous example, we would return the Q-value with a highest value, it could be the Q-value
    	# of the 'West' action or the Q-value of the 'North' action.
    	return max(q_values)

# *********************************************
    

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)


# *********************************************

# We will leave this part commented as we do not want to print it each time we train our agent.
#        print "Legal moves: ", legal
#        print "Pacman position: ", state.getPacmanPosition()
#        print "Ghost positions:" , state.getGhostPositions()
#        print "Food locations: "
#        print state.getFood()
#        print "Score: ", state.getScore()


# We will leave this part commented too as we do not want PAC-MAN to choose a random action.

        # Now pick what action to take. For now a random choice among
        # the legal moves
#        pick = random.choice(legal)

		
		# This corresponds to the second piece of the coursework: something that chooses how to act. This decision can be based on the
		# utilities, but also has to make sure that the learner does enough exploring.

        # Now we choose the next action:

        # We use the epsilon-greedy approach so the agent explores from time to time.
        random_number = random.random() # This generates a random float number in the range [0.0, 1.0)

        # 1. Sometimes we choose a random option to explore. Epsilon is the exploration rate.
        if random_number < self.epsilon:
            pick = random.choice(legal)

        # 2. The rest of the times, the vast majority of them, we choose the best option, the one that maximises the next Q-value. For 
        # this to work, we first have to see which are the next possible actions. We will store the Q-values in a util.Counter object,  
        # like before, so then we can get the key that has the highest value. The good thing is that this object has a method called 
        # argMax, that returns the key with the highest value, in other words, the action with the highest Q-value.
        else:

            # We initialise the Counter object
            q_values = util.Counter()

            # We store in the Counter object the Q-value (value) of each potential action (key):
            for action in legal:
                q_values[action] = self.q_table[(state, action)]

            # Now we choose the action (key) with the highest Q-value (value):
            pick = q_values.argMax()

		# Once we have the action already choosed, we take that action and observe the reward and the next state. Since the update of the
		# Q-value must be done after moving, before choosing the next move, we update the Q-value of the previous values of state, action
		# and reward. For this to work, we need to store the current action for the upgrade done in the next game step.

        # First, we store these values from the last state and action. These are the current actions and states, they will be the
        # previous state and action of the upgrade done in the next game step.
        lastState = self.s # Previous action.
        lastAction = self.a # Previous action.
        reward = state.getScore() - self.r # Reward obtained from taking that previous action in that previous state.

        # Then, we run the update. It is important to check if we are not in the initial state. If we were in the initial state,
        # lastState will be None and the program will return an error:
        if lastState != None:
            q_max = self.q_max(state) # Maximum Q-value for the given state: the next state if we think it from the point of view of the last state.
            self.update(lastState, lastAction, reward, q_max) # We perform the update.

        # Now we have to change again the future previous state, action and reward before taking the action:
        self.s = state # Next last state.
        self.a = pick # Next last action.
        self.r = state.getScore() # Next last score.
        
# *********************************************


        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):


# *********************************************

		# We have to make a last update of the Q-value when the game ends, when PAC-MAN either wins or loses. This must be done because
		# the getAction() function is not called once the game has ended. The last reward would not be stored to update the Q-values. As
		# it is stated in the coursework description, if the agent does not learn from these rewards, it may choose to run towards the 
		# the ghost to end the game quickly and minimise the losses.

        # First we store these values from taking the last state and action:
        lastState = self.s # Previous state.
        lastAction = self.a # Previous action.
        reward = state.getScore() - self.r # Reward obtained from taking that previous action.

        # Then, we run the update for the terminal state. q_max = 0 as there is no next state to maximise.
        self.update(lastState, lastAction, reward, 0) # We perform the update.


		# This does not end the program, it simply ends the game. Probably, more games will be played afterwards. Therefore, we have to
		# reset the values of state, action and score back to the ones they had in the constructor init() method. We do so:
        self.s = 0 # State.
        self.a = 0 # Action.
        self.r = 0 # Score.

# We will leave this part commented as we do not want to print it each time a game ends.
#        print "A game just ended!"

# *********************************************

      
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


# *********************************************

# BIBLIOGRAPHY: articles or webpages I used to elaborate this script

# Module slides.
# stackoverflow.com -> to resolve main programming doubts or to find better approaches to some problems.
# https://www.kdnuggets.com/2018/03/5-things-reinforcement-learning.html

# *********************************************
