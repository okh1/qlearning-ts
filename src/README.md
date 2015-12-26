This is the structure of the code:

- **QLearnerTABLE** is the class you should use if you want to use the classic QLearning algorithm which stores qvalues in a table. The table consists of rows where both the state features and the action are stored (the so called "qstate").
- **QLearnerNN** uses a neural network as a function approximator. Moreover, it uses some tricks like experience replay and reward clipping.
- **QLearnerBST** is the same as QLearnerTABLE, but a BST is used for the search function.

A crucial point is the transformation of the QState into a numerical representation as a vector, i.e. feature selection. You can try different features using the `QLearning.VectorMode` enum. Check the examples to see how it works.
