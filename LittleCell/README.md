----------- Little Cell --------------- 

Goal: Create a little cell in a 10x10 grid that can learn to survive by feeding itself on randomly distributed food in the top right corner

How I did it: I used expected sarsa to learn the value function of the 10x10 grid in an attempt to make the cell learn the overall location of food. I used an epsilon greedy policy

Why I failed: While the cell can survive and find out the general direction of food, it fails to accurately manage to gather enough food to survive a full life span. I might have not implemented the algorithm correctly which in turn made expected sarsa not work as desired. I might have not defined the model well enough. My rewards might have been a little bad, and I did not try every possible parameter.
 

What I learned: Expected Sarsa might not work to well in dynamic stochastic environments where the environment keeps continually changing, for example in this case what kept changing was the location of food. It was defined within a boundary but other than that it's spawn was random. Though this could also be due to poor implementation of the algorithm instead of the algorithm itself being insuficient. Anyhow, I gained a little bit more knowledge so this is good.
