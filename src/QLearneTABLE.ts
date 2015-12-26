/// <reference path="IQLearning.ts" />

module QLearning {

    export class QLearnerTABLE implements IQLearner {
        gamma: number; //Q-Learning parameter
        world: QLearning.World; //The world
        number_of_state_features: number; //Number of state features, i.e. length of state vector
        training: boolean; //True if we're learning, false if we're testing
        random: boolean; //True for epsilon-greedy policy, i.e. can choose action randomly sometimes
        epsilon: number; epsilon_min: number; //Parameters for epsilon-greedy policy
        t: number = 1; //Epoch number

        alfa: number; //Temporal difference learning rate
        learning_steps_total: number; //Used to calculate epsilon. After these number of steps, epsilon = epsilon_min
        start_learn_threshold: number; //When we should start learning
        learning_steps_burnin: number;
        //For table
        inputs: number[][];
        outputs: number[];

        constructor(world: QLearning.World, numberofstatefeatures: number) {
            //Initialize default parameters
            this.alfa = 0.7;
            this.gamma = 0.9;
            this.learning_steps_total = 500000;
            this.epsilon_min = 0;
            this.epsilon = 1;
            this.random = true;
            this.training = true;
            this.t = 1;
            this.start_learn_threshold = 1000;
            this.learning_steps_burnin = 500;
            //Assign user-defined parameters
            this.world = world; this.number_of_state_features = numberofstatefeatures;

            //Setup table where we store values
            var firstinput = [];
            for (var i = 0; i < this.number_of_state_features + this.world.num_actions; i++) {
                firstinput.push(0);
            }
            this.inputs = [firstinput]; this.outputs = [0];
        }

        getValue(state: QState, action: number) {
            var value: number;
            var oldaction = state.action;
            state.action = action;
            var index = this.indexOf(state);
            if (index != -1)
                value = this.outputs[index];
            else
                value = 0;
            state.action = oldaction;
            return value;
        }

        /* pi(s) = argmax_a Q(s, a)
           Returns the argmax over actions of Q(s, a).
           That is, returns the best action to take in state s according to our policy.
           It doesn't take not allowed action into consideration.
           Return value: {action: number, value: number} */
        private policy(state: QLearning.QState) {
            var action_values = new Array<number>(this.world.num_actions);
            for (var i = 0; i < this.world.num_actions; i++)
                action_values[i] = this.getValue(state, i);
            var maxk = 0;
            var maxval = action_values[0];
            for (var k = 1; k < this.world.num_actions; k++) {
                if (action_values[k] > maxval) { maxk = k; maxval = action_values[k]; }
            }
            return { action: maxk, value: maxval };
        }

        /* Method for choosing action either according to our policy or randomly in State state.
           Useful for exploration/exploitation.
           It doesn't take not allowed actions into considerations.
           Return value: an integer representing the action to take. */
        chooseAction(refstate: QLearning.QState) {
            //Epsilon-greedy
            this.epsilon = Math.min(1.0, Math.max(this.epsilon_min, 1.0 - (this.t - this.learning_steps_burnin) / (this.learning_steps_total - this.learning_steps_burnin)));
            var choose_randomly: boolean = this.random == true && Math.random() < this.epsilon; //Se scelgo random o no

            if (choose_randomly) { //Choose a random action
                return Utilities.getRandomInt(0, this.world.num_actions);
            }
            else { //Otherwise use our policy to make the decision
                var maxact = this.policy(refstate);
                return maxact.action;
            }
        }

        updateQ(oldstate: QLearning.QState, newstate: QLearning.QState, action: number, reward: number) {
            if (!this.training) return; else this.t++; //Increase iteration if we're learning

            var X = 0;

            if (!this.world.isfinalreward(reward)) {
                var qvalues = new Array<number>(this.world.num_actions);
                for (var i = 0; i < this.world.num_actions; i++)
                    qvalues[i] = this.getValue(newstate, i);
                X = reward + this.gamma * Math.max.apply(Math, qvalues);
            }
            else
                X = reward;

            var q = (1 - this.alfa) * this.getValue(oldstate, action) + this.alfa * X;

            var index: number = this.indexOf(oldstate);
            if (index != -1) {
                this.outputs[index] = q;
            }
            else {
                this.inputs.push(oldstate.ToVector()); this.outputs.push(q);
            }
        }

        private indexOf(state: QState) {
            var query = state.ToVector(); var found: boolean = false;
            for (var i = 0; i < this.inputs.length && !found; i++) {
                var flag: boolean = true;
                for (var j = 0; j < this.world.num_actions + this.number_of_state_features && flag; j++) {
                    if (this.inputs[i][j] != query[j])
                        flag = false;
                }
                if (flag) found = true;
            }
            if (found) return i - 1;
            else return -1;
        }

    }

}
