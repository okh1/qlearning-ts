interface IWorld {
    num_actions: number; //Number of possible actions

    /*  Method that takes the action stored in state and returns the new state.
        Alternatively, you can pass the newstate by reference. */
    makeaction(state: QLearning.QState, newstate?: QLearning.QState);

    /*  Method that calculates the reward of being in a state. */
    reward(state: QLearning.QState): number;

    /*  Method that checks if the action stored in state is allowed. */
    isactionpermitted(state: QLearning.QState): boolean;

    /*  Method that checks if the reward we got relates to a final state.*/
    isfinalreward(reward: number): boolean;
}

interface IQState {
    action: number; //Action of the QState. You should define an Action <-> integer correspondence.
    world: QLearning.World; //World we're playing in
    vector_mode: QLearning.VectorMode; //How to convert the state to a vector (e.g. for Pong all coordinate or the difference in height)

    /*  Returns a vector numerical representation of the state according to vector_mode. */
    ToVector(): number[];

    /*  Clone the current state. */
    clone(): QLearning.QState;

    /* Checks if the current state is final. */
    isfinalstate(): boolean;
}

interface IQLearner {
    gamma: number; //Q-Learning parameter
    world: QLearning.World; //The world
    number_of_state_features: number; //Number of state features, i.e. length of state vector
    training: boolean; //True if we're learning, false if we're testing
    random: boolean; //True for epsilon-greedy policy, i.e. can choose action randomly sometimes
    epsilon: number; //Parameter for epsilon-greedy policy: act randomly with epsilon probability
    epsilon_min: number; //Parameter for epsilon-greedy policy: minimum epsilon
    t: number; //Epoch number

    /*  Method for choosing action either according to our policy or randomly in State state.
        Useful for exploration/exploitation.
        It may or may not take not allowed actions into considerations.
        Return value: an integer representing the action to take. */
    chooseAction(state: QLearning.QState): number;

    /*  Method that implements the learning using the QLearning algorithm.
        When we are in state s_t+1 (newstate), we want to learn on the state s_t (oldstate),
        as we took action and got a reward. */
    updateQ(state: QLearning.QState, newstate: QLearning.QState, action: number, reward: number);

    /*  Returns Q(s,a). */
    getValue(state: QLearning.QState, action: number): number;
}
