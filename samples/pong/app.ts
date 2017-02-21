/// <reference path="Manager.ts" />
/// <reference path="QLearning.ts" />

var c, ctx; var width = 50; var height = 70; //For canvas
declare var $: any; //For jQuery
var interval; //JS timer
var trainingspeed = 10; //Timer interval in milliseconds. Smaller values = faster refresh = faster training
declare var convnetjs: any; declare var cnnutil: any; //convnetjs
var isdraw: boolean = false; var go: boolean = true; //They are true and false, respectively, to draw states
var t: number = 1; //Iteration number
var somma: number = 0; //Sum of scores
var greater: boolean = false; //True if a certain treshold for the score is crossed

var world: QLearning.World;
var state: QLearning.QState;
var learner_nn: QLearning.QLearnerNN = null;
var learner_table: QLearning.QLearnerTABLE = null;

window.onload = () => {
    setupchart();
    //Manage canvas, settubg its width and height
    c = document.getElementById('c');
    c.width = width; c.height = height;
    ctx = c.getContext('2d');
};

function core_nn() {
    var action = learner_nn.chooseAction(state);
    state.action = action;
    if (!world.isactionpermitted(state))
        state.action = 0;
    var newstate = world.makeaction(state);
    var reward = world.reward(newstate);
    learner_nn.updateQ(state, newstate, action, reward);
    updateScore(newstate);

    if (reward <= -1) {
        t++;
        somma += state.current_score;
        if (t % 1000 == 0) {
            console.log("I averaged " + somma / 1000 + " at iteration " + t + " with a cost of " + learner_nn.average_loss_window.sum
                + " with a reward of " + learner_nn.average_reward_window.sum);
            somma = 0;
        }
        newstate = new QLearning.QState(Utilities.getRandomInt(0, height - world.width_player - 1), Utilities.getRandomInt(world.radius_ball * 2, world.width - world.thickness_player * 2), Utilities.getRandomInt(world.radius_ball * 2, world.height - world.width_player * 2), 0,
            world, QLearning.VectorMode.Full); 
        newstate.speed_ball_x *= Utilities.getRandomOneOrMinusOne();
        newstate.speed_ball_y *= Utilities.getRandomOneOrMinusOne();
        newstate.highest_score = state.highest_score;
    }
    else {
        if (state.current_score > 10000) {
            console.log("GREATER THAN 10000! " + t);
            learner_nn.training = false;
        }
    }

    state = newstate;
    if (!go) draw(state);
}

function core_table() {
    var action = learner_table.chooseAction(state);
    state.action = action;
    var newstate = world.makeaction(state);
    var reward = world.reward(newstate);
    learner_table.updateQ(state, newstate, action, reward);
    updateScore(newstate);

    if (reward <= -1) {
        t++;
        somma += state.current_score;
        if (t % 10000 == 0) {
            console.log("I averaged " + somma / 10000 + " at iteration " + t );
            somma = 0;
        }
        newstate = new QLearning.QState(Utilities.getRandomInt(0, height - world.width_player - 1), Utilities.getRandomInt(world.radius_ball * 2, world.width - world.thickness_player * 2), Utilities.getRandomInt(world.radius_ball * 2, world.height - world.width_player * 2), 0,
            world, QLearning.VectorMode.DifferenceAction);
        newstate.speed_ball_x *= Utilities.getRandomOneOrMinusOne();
        newstate.speed_ball_y *= Utilities.getRandomOneOrMinusOne();
        newstate.highest_score = state.highest_score;
    }
    else {
        if (state.current_score > 10000) {
            console.log("GREATER THAN 10000! " + t);
            learner_table.training = false;
        }
    }
    state = newstate;

    if (!go) draw(state);
}


function startTrainingNN() {
    if (t == 1) {
        world = new QLearning.World(3, 10, 1, 20, 5, width, height);
        state = new QLearning.QState(Utilities.getRandomInt(0, height - world.width_player - 1), Utilities.getRandomInt(world.radius_ball * 2, world.width - world.thickness_player * 2), Utilities.getRandomInt(world.radius_ball * 2, world.height - world.width_player * 2), 0,
            world, QLearning.VectorMode.Full); //The starting state. Starting obstacles positions are 0 and 500 
        //Initialize network architecture
        //Difference-only
            //var layer_defs = [];
            //layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 });
            //layer_defs.push({ type: 'fc', num_neurons: 2, activation: 'relu' });
            //layer_defs.push({ type: 'regression', num_neurons: world.num_actions });
        //Full
        var layer_defs = [];
        layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 4 });
        layer_defs.push({ type: 'fc', num_neurons: 5, activation: 'relu' });
        layer_defs.push({ type: 'fc', num_neurons: 5, activation: 'relu' });
        layer_defs.push({ type: 'regression', num_neurons: world.num_actions });

        learner_nn = new QLearning.QLearnerNN(world, 4, layer_defs);

        //Difference-only
            //learner_nn.memory_size = 30000;
            //learner_nn.start_learn_threshold = 1000;
            //learner_nn.epsilon_min = 0;
            //learner_nn.gamma = 0.99;
            //learner_nn.learning_steps_total = 100000;
            //learner_nn.learning_steps_burnin = 1000;
            //learner_nn.use_final_reward = false;
            //learner_nn.reward_clipping = false;
            //world.reward_living = 0;
            //world.reward_win = 1;
            //world.reward_loss = -100;
        learner_nn.memory_size = 30000;
        learner_nn.start_learn_threshold = 1000;
        learner_nn.epsilon_min = 0.05;
        learner_nn.gamma = 0.99;
        learner_nn.learning_steps_total = 500000;
        learner_nn.learning_steps_burnin = 1000;
        learner_nn.trainer.momentum = 0.0;
        learner_nn.use_final_reward = true;
        learner_nn.reward_clipping = true;
        world.reward_living = 0.1;
        world.reward_win = 1;
        world.reward_loss = -1000;
    }

    window.clearInterval(interval);
    go = true;

    while (go) {
        core_nn();
    }
    interval = setInterval(core_nn, trainingspeed);
}

function startTrainingTable() {
    if (t == 1) {
        world = new QLearning.World(3, 10, 1, 20, 5, width, height);
        state = new QLearning.QState(Utilities.getRandomInt(0, height - world.width_player - 1), Utilities.getRandomInt(world.radius_ball * 2, world.width - world.thickness_player * 2), Utilities.getRandomInt(world.radius_ball * 2, world.height - world.width_player * 2), 0,
            world, QLearning.VectorMode.DifferenceAction); //The starting state. Starting obstacles positions are 0 and 500 

        learner_table = new QLearning.QLearnerTABLE(world, 2);
        learner_table.random = false;
        world.reward_living = 0;
        world.reward_win = 100;
        world.reward_loss = -1000;
    }

    window.clearInterval(interval);
    go = true;
    while (go)
        core_table();
    isdraw = true;
    interval = setInterval(core_table, trainingspeed);
}