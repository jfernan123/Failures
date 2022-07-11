import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
pygame.display.set_caption("Physics")


class Grid:
    def __init__(self, screenWidth, screenHeight, gridWidth, gridHeight, cell):
        self.gridHeight = gridHeight
        self.gridWidth = gridWidth
        self.statsWidth = 300
        self.width, self.height = screenWidth, screenHeight
        self.recHeight = self.width // self.gridWidth
        self.recWidth = self.height // self.gridHeight
        self.borderSize = 1
        self.size = self.gridHeight * self.gridWidth
        self.grid_blocks = []
        self.grid_blocksCoordinates = []
        self.foodLoc = None
        self.foodSpawned = False
        self.statsDrawn = False
        self.cell = cell

    def show(self):
        pygame.init()
        screen = (self.width + self.statsWidth, self.height)
        # Initiate screen
        screenSurface = pygame.display.set_mode(screen)
        running = True
        clock = pygame.time.Clock()

        # Draw the current grid
        # Run the game loop
        self.drawGrid(screenSurface)

        while running:
            self.updateGridState()
            self.drawCurrentState(screenSurface)

            pygame.display.update()

    def drawCurrentState(self, screenSurface, clock, fps):
        clock.tick(fps)
        screenSurface.fill(pygame.Color("Black"), (self.width,
                           0, self.width+self.statsWidth, self.height))
        self.drawCell(self.cell, screenSurface)
        self.drawFood(screenSurface)
        self.drawStats(screenSurface)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

    def updateGridState(self):
        # action = random.randrange(0,4)
        # self.cell.move(action)
        self.spawnFood()
        if self.cell.eat(self.foodLoc):
            # Eat it.
            self.foodSpawned = False

    def drawGrid(self, screenSurface):
        for row in range(self.gridHeight):
            for column in range(self.gridWidth):
                # Get the width and height of each rectangle and border size
                # Draw the rectangle
                currentRec = pygame.Rect(column * self.recWidth, row * self.recHeight,
                                         self.recWidth - self.borderSize, self.recHeight - self.borderSize)
                pygame.draw.rect(
                    screenSurface, pygame.Color("white"), currentRec)
                self.grid_blocks.append(currentRec)

    def restoreDefaults(self, screenSurface):
        screenSurface.fill(pygame.Color("Black"))
        self.drawGrid(screenSurface)
        self.foodSpawned = False

    def drawCell(self, cell, screenSurface):
        # Get code of little cell
        prevLoc = cell.getPrevLoc()

        location = cell.getCode()
        # Get current rectangle of the cell and draw rectangle
        cellRec = self.grid_blocks[location].copy()

        pygame.draw.rect(screenSurface, pygame.Color("Red"), cellRec)

        if prevLoc != None:
            prevRec = self.grid_blocks[prevLoc].copy()
            pygame.draw.rect(screenSurface, pygame.Color("White"), prevRec)

    def spawnFood(self):
        # Pick a random location between the range .

        if not self.foodSpawned:
            randomNumX = random.uniform(0, 1)
            randomNumY = random.uniform(0, 1)
            if randomNumX < 0.25:
                spawnX = (self.gridWidth - 1) - 4
            elif randomNumX < 0.75:
                spawnX = random.choice(
                    [(self.gridWidth - 1) - 3, (self.gridWidth - 1) - 2])
            else:
                spawnX = self.gridWidth-1

            if randomNumY < 0.25:
                spawnY = 0
            elif randomNumY < 0.75:
                spawnY = random.choice(
                    [1, 2])
            else:
                spawnY = 3
            # If the cell collides with the location then make it into food.
            self.foodLoc = self.getLocCode(spawnX, spawnY)
            self.foodSpawned = True

    def drawFood(self, screenSurface):
        foodRec = self.grid_blocks[self.foodLoc].copy()
        pygame.draw.rect(screenSurface, pygame.Color("Green"), foodRec)

    def getLocCode(self, x, y):
        return self.gridWidth * y + x

    def drawStats(self, screenSurface):

        energyStr = "Energy: " + str(self.cell.getEnergyLevel())
        lifeStr = "Life Remaining: " + str(self.cell.getLifeSpan())
        font = pygame.font.Font(pygame.font.get_default_font(), 20)

        energyText = font.render(energyStr, False, "White")
        lifeSpanText = font.render(lifeStr, False, "White")

        screenSurface.blit(lifeSpanText, (self.width + 5, 300))
        screenSurface.blit(energyText, (self.width + 5, 0))

    def getTotalSize(self):
        return self.size


class Cell:
    def __init__(self, gridWidth, gridHeight):
        self.x = 0
        self.y = 0
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.actionUP = 0
        self.actionDOWN = 1
        self.actionLEFT = 2
        self.actionRIGHT = 3
        self.actionSTILL = 4
        self.lifeSpan = 200
        self.energyLevel = 50
        self.totalLifeSpan = self.lifeSpan
        self.totalEnergy = self.energyLevel
        self.numActions = 4
        self.prevLoc = None
        self.recentlyAte = False

    def getLocation(self):
        return (self.x, self.y)

    def getCode(self):
        return self.gridWidth * self.y + self.x

    def move(self, action):
        # There are 5 actions namely 0 = up, 1 = down, 2 = left, 3 = right, 4 = stay still
        prevPrevLoc = self.prevLoc
        self.prevLoc = self.getCode()

        prevX = self.x
        prevY = self.y
        if action == self.actionUP:
            self.y = self.y - 1
        elif action == self.actionDOWN:
            self.y = self.y + 1
        elif action == self.actionLEFT:
            self.x = self.x - 1
        elif action == self.actionRIGHT:
            self.x = self.x + 1
        self.energyLevel -= 1
        self.lifeSpan -= 1
        # Check if it's in bounds. If it's not in bounds return to previous X and Y
        if not self.__InBounds():
            self.x = prevX
            self.y = prevY
            self.prevLoc = prevPrevLoc
        self.energyLevel -= 0

    def getPrevLoc(self):
        return self.prevLoc

    def __InBounds(self):
        if self.x < 0 or self.x >= self.gridWidth:
            return False
        elif self.y < 0 or self.y >= self.gridHeight:
            return False
        else:
            return True

    def fullBelly(self):
        ate = self.recentlyAte
        self.recentlyAte = False
        return ate

    def eat(self, foodCode):
        # If the cell has eaten it will return true, otherwise false.
        if self.getCode() == foodCode:
            energyToAdd = 40
            if self.energyLevel + energyToAdd > self.totalEnergy:
                self.energyLevel = self.totalEnergy
            else:
                self.energyLevel += energyToAdd
            self.recentlyAte = True
            return True
        else:
            return False

    def getLifeSpan(self):
        return self.lifeSpan

    def getTotalLifeSpan(self):
        return self.totalLifeSpan

    def getTotalEnergy(self):
        return self.totalEnergy

    def getEnergyLevel(self):
        return self.energyLevel

    def getNumberOfActions(self):
        return self.numActions

    def outOfEnergy(self):
        return self.energyLevel <= 0

    def diedOfOldAge(self):
        return self.lifeSpan <= 0
    # Move agent up or down

    def restoreDefaults(self):
        self.lifeSpan = self.totalLifeSpan
        self.energyLevel = self.totalEnergy
        self.x = 0
        self.y = 0
        self.prevLoc = None
# A policy is the probability of taking an action given a current state.

# Initially each action will have an equal likelihood of happening.

# Then through Q-Learning I will learn the optimal probability for each action state

# Firstly we need a way to figure out how to start each episode

# Then we need a way to advance each time step


# Get state

# I need to find a way to map each unique combination to a different numbered state.

# And I need to be capable of obtaining that mapping back.

# Location + Energy Level +  Life Span

# Action = policy[location][Energy Level][LifeSpan][action]
# 100 locations 10 energy levels 50 lifespan 4 different actions for each given state.
# 2 Locations 2 energy levels 2 life span


# In this case I will use an epsilon greedy policy with expected sarsa.


# I based this structure of this class on the RL Glue protocol which I saw on the coursera course "Reinforcement Learning with specialization".
class ExpectedSarsaAgent:
    # Step size
    # Discount
    # Value function
    def agent_init(self, Cell, theGrid):

        self.epsilon = 0.05
        self.step_size = 0.5
        self.discount = 0.5
        self.cell = Cell
        self.theGrid = theGrid
        self.q_values = np.zeros((self.theGrid.getTotalSize(), self.cell.getTotalEnergy(
        ), self.cell.getTotalLifeSpan(), self.cell.getNumberOfActions()))

    def agent_start(self, observation):

        # In each episode start an agent in a random location in the map

        # Select a random move based on the epsilon greedy policy

        state = observation

        # All q values are initiated to 0 so the all actions are greedy regardless. Thus their chance is equal.
        # Action = q_values[location][Energy Level][LifeSpan][action]

        # Select the greedy action
        prob = random.uniform(0, 1)
        location = state[0]
        energyLevel = state[1]
        lifeSpan = state[2]
        currentStateVals = self.q_values[location][energyLevel][lifeSpan]
        if prob < self.epsilon:
            action = random.randrange(self.cell.getNumberOfActions() - 1)
        else:
            action = self.argMax(currentStateVals)
        # Move the cell to our desired action
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, observation):

        # Observation - Vector containing location, energy leven, and life span remaining.
        # Reward - Reward obtained during the current state.

        # State observed
        state = observation
        prob = random.uniform(0, 1)
        location = observation[0]
        energyLevel = observation[1]
        lifeSpan = observation[2]
        currentStateVals = self.q_values[location][energyLevel][lifeSpan]
        if prob < self.epsilon:
            action = random.randrange(self.cell.getNumberOfActions())
        else:
            action = self.argMax(currentStateVals)

        expected = 0
        count = 0
        numActions = self.cell.getNumberOfActions()
        for myAction in currentStateVals:

            if myAction == max(currentStateVals) and count < 1:
                expected += (1 - self.epsilon +
                             (self.epsilon / numActions)) * myAction
                count += 1
            else:
                expected += (self.epsilon / numActions) * myAction
        prevLoc = self.prev_state[0]
        prevEnergyLevel = self.prev_state[1]
        prevLifeSpan = self.prev_state[2]
        prevActionValue = self.q_values[prevLoc][prevEnergyLevel][prevLifeSpan][self.prev_action]
        self.q_values[prevLoc][prevEnergyLevel][prevLifeSpan][self.prev_action] = (prevActionValue +
                                                                                   self.step_size *
                                                                                   (reward + (self.discount*expected) - prevActionValue))
        self.prev_state = state
        self.prev_action = action
        return action
        # Select a random action

    def agent_end(self, reward):
        prevLoc = self.prev_state[0]
        prevEnergyLevel = self.prev_state[1]
        prevLifeSpan = self.prev_state[2]
        prevActionValue = self.q_values[prevLoc][prevEnergyLevel][prevLifeSpan][self.prev_action]
        self.q_values[prevLoc][prevEnergyLevel][prevLifeSpan][self.prev_action] = (prevActionValue +
                                                                                   self.step_size*(reward - prevActionValue))

    def clean_up(self, screenSurface):
        self.theGrid.restoreDefaults(screenSurface)

        self.cell.restoreDefaults()

    def argMax(self, actionList):
        maxIndex = 0
        ties = []
        for action in range(len(actionList)):
            if actionList[action] > actionList[maxIndex]:
                maxIndex = action
                ties = [action]
            else:
                ties.append(action)
        return random.choice(ties)


def main():
    gridWidth = 10
    gridHeight = 10
    screenWidth = 800
    screenHeight = 800
    statsWidth = 300
    fps = -1
    numEpisodes = 5000000

    pygame.init()

    cell = Cell(gridWidth, gridHeight)

    ourGrid = Grid(screenWidth, screenHeight, gridWidth, gridHeight, cell)

    cellSarsa = ExpectedSarsaAgent()
    cellSarsa.agent_init(cell, ourGrid)
    # The time steps depend on the life span and the energy level
    startVector = (0, cell.getTotalEnergy() - 1, cell.getTotalLifeSpan() - 1)
    # Initiate screen

    clock = pygame.time.Clock()

    screen = (screenWidth + statsWidth, screenHeight)
    screenSurface = pygame.display.set_mode(screen)
    running = True
    ourGrid.drawGrid(screenSurface)
    episodeReward = {}
    for i in tqdm(range(numEpisodes)):
        action = cellSarsa.agent_start(startVector)
        cell.move(action)
        while not cell.outOfEnergy() and not cell.diedOfOldAge():

            ourGrid.updateGridState()
            if (i + 1) / numEpisodes > 1:

                ourGrid.drawCurrentState(screenSurface, clock, fps)

            if cell.diedOfOldAge():
                reward = 1000000
                cellSarsa.agent_end(reward)
            elif cell.outOfEnergy():
                reward = -1000
                cellSarsa.agent_end(reward)
            elif cell.fullBelly():
                reward = 100
                observation = [
                    cell.getCode(), cell.getEnergyLevel() - 1, cell.getLifeSpan() - 1]
                action = cellSarsa.agent_step(reward, observation)
                cell.move(action)
            else:
                reward = -1
                observation = [
                    cell.getCode(), cell.getEnergyLevel() - 1, cell.getLifeSpan() - 1]
                action = cellSarsa.agent_step(reward, observation)
                cell.move(action)
            pygame.display.update()
        cellSarsa.clean_up(screenSurface)
        episodeReward[i] = np.sum(cellSarsa.q_values)
        if (i + 1) / numEpisodes > 0.999:
            fps = 15
    # As the cell could die within 10 time steps.
    # sorted by key, return a list of tuples
    np.save('10Hour_Training', cellSarsa.q_values)

    lists = sorted(episodeReward.items())

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.show()

    # Thus the life span

    # Let's try a few episodes
    print(cellSarsa.q_values)


[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

# screenWidth = 500
# screenHeight = 500
# gridWidth = 10
# gridHeight = 10
# grid = Grid(screenWidth, screenHeight, gridWidth, gridHeight)
# grid.show()
main()
