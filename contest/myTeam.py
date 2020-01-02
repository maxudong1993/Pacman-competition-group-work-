# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'QLearningDefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

  def getSuccessor(self, gameState, action):
    #find the next state
    successor = gameState.generateSuccessor(self.index,action)
    position = successor.getAgentState(self.index).getPosition()
    if position != nearestPoint(position):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

class QLearningDefensiveAgent(DummyAgent):
  
  def initParameters(self, epsilon = 0.05, alpha = 0.5, gamma = 0.7):
    '''
    epsilon -> exploration rate (epsilon-greedy) 
    alpha -> learning rate
    gamma -> discount factor
    '''

    self.epsilon = float(epsilon)
    self.alpha = float(alpha)
    self.discount = float(gamma)
    self.qTable = util.Counter()

  def registerInitialState(self, gameState):
    self.initParameters()
    DummyAgent.registerInitialState(self,gameState)

  def chooseAction(self,gameState):

    actions = gameState.getLegalActions(self.index)
    
    for action in actions:
      self.updateQ(gameState,action) 

    result = self.getBestAction(gameState)
    if util.flipCoin(self.epsilon):
      return random.choice(actions)
    else:
      #print "qLearning"
      #print self.qTable[(gameState.getAgentState(self.index).getPosition(),result)]
      return result

  def getQValue(self, position, action):
    return self.qTable[(position,action)]

  def getValue(self, gameState):

    #return the max Q(s,a) of a state s
    position = gameState.getAgentState(self.index).getPosition()
    actions = gameState.getLegalActions(self.index)
    qValues = []

    for action in actions:
      qValues.append(self.getQValue(position,action))

    if len(actions) == 0:
      return 0.0
    else:
      return max(qValues)

  def updateQ(self, gameState, action):

    currentPosition = gameState.getAgentState(self.index).getPosition()

    successor = self.getSuccessor(gameState, action)
    nextState = successor.getAgentState(self.index)
    nextPosition = nextState.getPosition()

    #used to define a reward function
    rewardFeatures = util.Counter()

    
    rewardFeatures['isDefender'] = 1
    if nextState.isPacman: rewardFeatures['isDefender'] = 0

    opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemies = [enemy for enemy in opponents if enemy.isPacman and enemy.getPosition() != None]
    rewardFeatures['numOfEnemies'] = len(enemies)

    '''
    if(nextState.scaredTimer != 0):
      print "white  ***:" + str(nextState.scaredTimer)
    '''

    if len(enemies) > 0:
      #print 'distance **********'
      distances = [self.getMazeDistance(nextPosition, enemy.getPosition()) for enemy in enemies]
      rewardFeatures['enemyDistance'] = min(distances)

    if action == Directions.STOP: rewardFeatures['stop'] = 1
    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == reverse: rewardFeatures['reverse'] = 1

    rewardWeight = {'numOfEnemies': -1000, 'isDefender': 100, 'enemyDistance': -500, 'stop': -100, 'reverse': -200}
    reward = rewardFeatures * rewardWeight

    '''
    print 'numOfEnemies ' +str(rewardWeight['numOfEnemies']*rewardFeatures['numOfEnemies'])
    print 'isDefender ' +str(rewardWeight['isDefender']*rewardFeatures['isDefender'])
    print 'enemyDistance ' +str(rewardWeight['enemyDistance']*rewardFeatures['enemyDistance'])
    print 'stop ' +str(rewardWeight['stop']*rewardFeatures['stop'])
    print 'reverse ' +str(rewardWeight['reverse']*rewardFeatures['reverse'])
    '''

    if len(successor.getLegalActions(self.index)) == 0:
      differ = reward
    else:
      differ = reward + (self.discount * self.getValue(successor))

    factorOne = (1 - self.alpha) * self.getQValue(currentPosition,action)
    factorTwo = self.alpha * differ
    #update Q table
    self.qTable[(currentPosition,action)] = factorOne + factorTwo
    #print str(currentPosition) +" " +str(action) +" " +str(self.qTable[(currentPosition,action)])

  def getBestAction(self, gameState):

    currentPosition = gameState.getAgentState(self.index).getPosition()
    maxQValue = self.getValue(gameState)
    actions = gameState.getLegalActions(self.index)
    bestActions = [action for action in actions if self.getQValue(currentPosition,action) == maxQValue] 
    return random.choice(bestActions)


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman == False and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      distanceToGhost = min(dists)
      if distanceToGhost <= 5:
         features['distanceToGhost'] = - distanceToGhost * len(foodList)
      #print(min(dists))
    else:
      features['distanceToGhost'] = 0
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1,'distanceToGhost':2000} 
