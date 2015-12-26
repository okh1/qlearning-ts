/// <reference path="IQLearning.ts" />

module QLearning {

    /*  Learner that uses a neural network as a function approximator.
        It uses replay memory as described by Google DeepMind.
        The neural network library used is convnetjs. */
    export class QLearnerNN implements IQLearner {
        gamma: number; //Q-Learning parameter
        random: boolean; //True for epsilon-greedy policy, i.e. can choose action randomly sometimes
        world: QLearning.World; //The world
        number_of_state_features: number; //Number of state features, i.e. length of state vector
        mode: string; //table or nn or hash or bst
        training: boolean; //True if we're learning, false if we're testing
        epsilon: number; epsilon_min: number; //Parameters for epsilon-greedy policy
        t: number = 1; //Epoch number

        learning_steps_total: number; //Used to calculate epsilon. After these number of steps, epsilon = epsilon_min
        start_learn_threshold: number; //When we should start learning
        learning_steps_burnin: number;
        memory: Array<Experience>; memory_size: number; //Replay memory
        average_loss_window: any; //Cost, error in predicting qvalues by neural network
        average_reward_window: any;
        layer_defs; net; trainer; //For convnetjs
        use_final_reward: boolean; //If true, final states will learn the undiscounted reward
        reward_clipping: boolean; //If true, rewards are scaled

        constructor(world: QLearning.World, numberofstatefeatures: number, layer_defs?: any, opt_trainer?: any) {
            //Initialize default parameters
            this.gamma = 0.99;
            this.learning_steps_total = 500000;
            this.epsilon_min = 0;
            this.epsilon = 1;
            this.random = true;
            this.training = true;
            this.t = 1;
            this.start_learn_threshold = 1000;
            this.learning_steps_burnin = 1000;
            this.use_final_reward = true;
            this.reward_clipping = false;
            //Assign user-defined parameters
            this.world = world; this.number_of_state_features = numberofstatefeatures;

            //Setup neural network
            //Initialize parameters
            this.memory_size = 30000;
            this.average_loss_window = new cnnutil.Window(1000, 10);
            this.average_reward_window = new cnnutil.Window(1000, 10);
            //Initialize network
            if (layer_defs == null) { //Default network structure
                this.layer_defs = [];
                this.layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: this.number_of_state_features });
                this.layer_defs.push({ type: 'fc', num_neurons: 2, activation: 'relu' });
                this.layer_defs.push({ type: 'fc', num_neurons: 2, activation: 'relu' });
                this.layer_defs.push({ type: 'regression', num_neurons: this.world.num_actions });
            }
            else
                this.layer_defs = layer_defs;
            this.net = new convnetjs.Net();
            this.net.makeLayers(this.layer_defs);
            if (opt_trainer == null) //Default trainer options
                opt_trainer = { learning_rate: 0.001, batch_size: 64, l2_decay: 0.01, momentum: 0 };
            this.trainer = new convnetjs.SGDTrainer(this.net, opt_trainer);
            //Initialize replay memory
            this.memory = new Array<Experience>();
        }

        /*  This is where we learn. We update our replay memory, then take a batch of experiences and train
            the neural network on them using the QLearning equation Q(s,a) = r + max_overactions Q(s',a). */
        updateQ(oldstate: QLearning.QState, newstate: QLearning.QState, action: number, reward: number) {
            if (!this.training) return; else this.t++; //Increase iteration if we're learning

            //Update replay memory
            if (this.memory.length < this.memory_size) { //Add new
                var xp = new Experience(oldstate.ToVector(), newstate.ToVector(), action, reward);
                this.memory.push(xp);
            }
            else { //Replace existing
                var ri = Utilities.getRandomInt(0, this.memory_size);
                var xp = new Experience(oldstate.ToVector(), newstate.ToVector(), action, reward);
                this.memory[ri] = xp;
            }
            //Train batch
            if (this.memory.length > this.start_learn_threshold) {
                var cost: number = 0;
                for (var i = 0; i < this.trainer.batch_size; i++) {
                    var ri = Utilities.getRandomInt(0, this.memory.length);
                    var xp: Experience = this.memory[ri];
                    var state = new convnetjs.Vol(1, 1, this.number_of_state_features);
                    state.w = xp.state0;
                    var maxact = this.policy(xp.state1);
                    if (isNaN(maxact.value))
                        console.log("nan");
                    var value;
                    if (!this.use_final_reward || !this.world.isfinalreward(xp.reward0))
                        value = xp.reward0 + this.gamma * this.decode(maxact.value);
                    else
                        value = xp.reward0;
                    var ystruct = { dim: xp.action0, val: this.encode(value) };
                    var loss = this.trainer.train(state, ystruct);
                    cost += loss.loss;
                }
                cost = cost / this.trainer.batch_size;
                this.average_loss_window.add(cost);
            }
            this.average_reward_window.add(reward);
        }

        chooseAction(refstate: QLearning.QState) {
            var state = refstate.ToVector();
            //Epsilon-greedy
            this.epsilon = Math.min(1.0, Math.max(this.epsilon_min, 1.0 - (this.t - this.learning_steps_burnin) / (this.learning_steps_total - this.learning_steps_burnin)));
            var choose_randomly: boolean = this.random == true && Math.random() < this.epsilon; //Se scelgo random o no

            if (choose_randomly) { //Choose a random action
                return Utilities.getRandomInt(0, this.world.num_actions);
            }
            else { //Otherwise use our policy to make the decision
                var maxact = this.policy(state);
                if (isNaN(maxact.value))
                    console.log("nan");
                return maxact.action;
            }
        }

        /*  Returns Q(s, a), that is the value of being in state s and doing
            action a. */
        getValue(state: QLearning.QState, action: number) {
            var state_vol = new convnetjs.Vol(1, 1, this.number_of_state_features);
            state_vol.w = state.ToVector();
            var action_values = this.net.forward(state_vol);
            return action_values.w[action];
        }

        /* pi(s) = argmax_a Q(s, a)
           Returns the argmax over actions of Q(s, a).
           That is, returns the best action to take in state s according to our policy.
           It doesn't take not allowed action into consideration.
           Return value: {action: number, value: number} */
        private policy(state: number[]) {
            var state_vol = new convnetjs.Vol(1, 1, this.number_of_state_features);
            state_vol.w = state;
            var action_values = this.net.forward(state_vol);
            var maxk = 0;
            var maxval = action_values.w[0];
            for (var k = 1; k < this.world.num_actions; k++) {
                if (action_values.w[k] > maxval) { maxk = k; maxval = action_values.w[k]; }
            }
            return { action: maxk, value: maxval };
        }

        /* Scale the reward to feed it to the network. */
        private encode(x: number) {
            if (!this.reward_clipping)
                return x;
            else {
                var ymax = -1000; var ymin = 1; var xmin = -1000; var xmax = 1;
                return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin;
            }
        }

        /* Descale the reward to use it. */
        private decode(x: number) {
            if (!this.reward_clipping)
                return x;
            else {
                var ymax = -1000; var ymin = 1; var xmin = -1; var xmax = 1;
                return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin;
            }
        }

    }


    /*  Useful class for storing experience as a tuple (s, a, r, s').
        Used in replay memory. */
    class Experience {
        state0: number[];
        action0: number;
        state1: number[];
        reward0: number;

        constructor(old_state: number[], new_state: number[], action: number, reward: number) {
            this.state0 = old_state;
            this.state1 = new_state;
            this.action0 = action;
            this.reward0 = reward;
        }
    }
}
