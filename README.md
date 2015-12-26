# qlearning-ts
This is a TypeScript implementation of the QLearning algorithm for Reinforcement Learning you can execute in your browser. It supports:

- Classic QLearning with values stored in a **table**
- QLearning with values stored in a **Binary Search Tree** for faster search
- QLearning with a **neural network** as function approximator (someone calls this Deep Q Learning). The Javascript neural network library used is convnetjs, but you can easily adapt the code to use your own library.

You can check the interface `IQLearning` to see which functions are mandatory to make it working. You have three basic piecies:

- the **World**, which describes the rules of the game we're playing
- the **State**, which represents each game state
- the **Learner**, where the learning happens using the QLearning algorithm.

## Example code
Once you've implemented every function, and initialized the variables `world`, `state` and `learner`, the process of learning is straightforward to code:

```javascript
var action = learner.chooseAction(state);
state.action = action;
var newstate = world.makeaction(state);
var reward = world.reward(newstate);
learner.updateQ(state, newstate, action, reward);
state = newstate;
```

You should repeat this until the agent has learnt how to win. You may change this sample code in a lot of ways, depending on your needs. For example, you can check if the action chosen by the agent is allowed, or you can reset the state once you've reached some sort of final state.
