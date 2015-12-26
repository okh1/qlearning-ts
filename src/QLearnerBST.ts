/// <reference path="IQLearning.ts" />

module QLearning {

    export class QLearnerBST implements IQLearner {
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
        //For bst
        tree: BST.IndexedBST<number[]>;

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
            //Setup tree
            this.tree = new BST.IndexedBST(this.inputs, this.compare);
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
                this.tree.insert(oldstate.ToVector(), this.tree.N + 1); //Il più 1 è perché il primo elemento di inputs è quello nullo
            }
        }

        /* Returns the index of state in the inputs array */
        private indexOf(state: QState) {
            var query = state.ToVector();
            return this.tree.search(query);
        }

        private compare(a: number[], b: number[]) {
            var length = a.length;

            for (var i = 0; i < length; i++) {
                if (a[i] < b[i])
                    return -1;
                else if (a[i] > b[i])
                    return 1;
            }
            return 0;
        }
    }

}

module BST {
    /* <0 means a is smaller, = 0 means they are equal, >0 means a is larger */
    interface ICompareFunction<T> {
        (a: T, b: T): number;
    }

    export class BSTNode {
        index: number;
        left: BSTNode;
        right: BSTNode;

        constructor(index: number, left: BSTNode, right: BSTNode) {
            this.index = index;
            this.left = left;
            this.right = right;
        }
    }

    export class IndexedBST<T> {
        private root: BSTNode; //Tree's root
        private end_node: BSTNode; //For leaves
        N: number; //Number of elements
        private arr: Array<T>; //Referenced array, where data is
        private compare: ICompareFunction<T>; //Compare function

        constructor(array: Array<T>, compare: ICompareFunction<T>) {
            this.N = 0;
            this.end_node = new BSTNode(-1, null, null);
            this.arr = array;
            this.compare = compare;
        }

        search(element: T) {
            if (this.N == 0)
                return -1;
            else
                return this.search_in_tree(element, this.root);
        }

        private search_in_tree(element: T, head: BSTNode) {
            if (head == this.end_node) //Sono arrivato alla fine
                return -1;
            var comparision = this.compare(element, this.arr[head.index]);
            if (comparision == 0)
                return head.index;
            else if (comparision < 0) //Search in left subtree
                return this.search_in_tree(element, head.left);
            else if (comparision > 0) //Search in right subtree
                return this.search_in_tree(element, head.right);
        }

        insert(element: T, index: number) {
            if (this.N > 0)
                this.insert_in_tree(element, index, this.root);
            else
                this.root = new BSTNode(index, this.end_node, this.end_node);
            this.N++;
        }

        private insert_in_tree(element: T, index: number, head: BSTNode) {
            if (head == this.end_node) {
                return new BSTNode(index, this.end_node, this.end_node);
            }
            var comparision = this.compare(element, this.arr[head.index]);
            if (comparision < 0) //Insert in left subtree
                head.left = this.insert_in_tree(element, index, head.left);
            else if (comparision > 0) //Insert in right subtree
                head.right = this.insert_in_tree(element, index, head.right);
            return head;
        }
    }
}
