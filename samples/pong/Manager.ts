//CODE FOR PLOTTING CHARTS, DRAWING STATES AND UPDATING SCORE

declare var Dygraph: any; declare var g: any; var data = []; var ascissa = 0; var plot = false; //For chart
declare var g2: any; var data2 = []; var ascissa2 = 0; //For error chart

//Computes the values for Action action in State state
function nnet_brain(state: QLearning.QState, action: number, brain) {
    var vector = new convnetjs.Vol(state.ToVector());
    var results = brain.tdtrainer.net.forward(vector, true);
    return results.w[action];
}

//Draws the current state to the canvas
function draw(state: QLearning.QState) {
    //Clear canvas
    ctx.clearRect(0, 0, state.world.width, state.world.height);
    //Draw ball
    ctx.fillStyle = 'grey';
    ctx.fillRect(state.ball_x, state.world.height - state.ball_y - state.world.radius_ball, state.world.radius_ball, state.world.radius_ball);

    //Draw player
    ctx.fillStyle = 'black';
    ctx.fillRect(state.world.width - state.world.thickness_player, state.world.height - state.y - state.world.width_player, state.world.thickness_player, state.world.width_player);
}

//Updates the score in the HTML page and in the state
function updateScore(state: QLearning.QState) {
    if (state.isfinalstate())
        state.current_score = 0;
    else
        state.current_score++;
    if (state.current_score > state.highest_score) state.highest_score = state.current_score;
    if (isdraw) {
        $("#spanScore").text(state.current_score.toString());
        $("#spanHighestScore").text(state.highest_score.toString());
    }
}

//Chart methods
function setupchart() {
    //Set up della chart che mi dice il valore di ogni azione
    for (var i = 100; i >= 0; i--) {
        ascissa++;
        var y = Math.random();
        data.push([ascissa, y, y + 1, y - 1]);
    }


    g = new Dygraph(document.getElementById("div_g"), data,
        {
            drawPoints: true,
            labels: ['t', 'Nothing', 'Up', 'Down']
        });

}

function plotchart(s: QLearning.QState) {
    //Aggiunge un data point al primo chart
    ascissa++;
    var clonestate: QLearning.QState = s.clone();
    //var v1 = nnet_brain(clonestate, 0);
    //var v2 = nnet_brain(clonestate, 1);
    //var v3 = nnet_brain(clonestate, 2);

    data.shift();
    //data.push([ascissa, v1, v2, v3]);
    g.updateOptions({ 'file': data });
}

function setupchart2() {
    //Set up del chart che mi dà l'errore del neural network
    for (var i = 100; i >= 0; i--) {
        ascissa2++;
        var y = Math.random();
        data2.push([ascissa2, y]);
    }


    g2 = new Dygraph(document.getElementById("div_g2"), data,
        {
            drawPoints: true,
            labels: ['t', 'Error']
        });

}

function ploterror(error: number) {
    ascissa2++;

    data2.shift();
    data2.push([ascissa, error]);
    g2.updateOptions({ 'file': data2 });
}
