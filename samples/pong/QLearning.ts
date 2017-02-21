/// <reference path="QLearning\IQLearning.ts" />

module QLearning {
    enum Action { Do_Nothing, Up, Down };
    export enum VectorMode { Difference, Full, DifferenceAction };

        export class World implements IWorld {
            num_actions: number;

            speed_player: number; //Speed of player
            width_player: number;
            thickness_player: number = 5;
            radius_ball: number;
            speed_ball: number; //Speed of ball
            width: number; height: number;
            reward_loss: number = -100;
            reward_win: number = 1;
            reward_living: number = 0.1;

            constructor(number_of_actions: number, speed_of_player: number, speed_of_ball: number, width_of_player: number, radius_of_ball: number, width: number, height: number) {
                this.num_actions = number_of_actions;
                this.speed_player = speed_of_player;
                this.speed_ball = speed_of_ball;
                this.width_player = width_of_player;
                this.radius_ball = radius_of_ball;
                this.width = width; this.height = height;
            }

            makeaction(refstate: QState) {
                var state = refstate.clone();
                if (!this.isactionpermitted(state))
                    state.action = Action.Do_Nothing;
                //Move player
                if (state.action == Action.Up)
                    state.y += this.speed_player;
                else if (state.action == Action.Down)
                    state.y -= this.speed_player;
                //Move ball
                if (state.ball_x <= 0) {
                    state.speed_ball_x *= -1;
                    state.ball_x += state.speed_ball_x;
                }
                else if (state.ball_y + state.world.radius_ball >= height || state.ball_y <= 0) {
                    state.speed_ball_y *= -1;
                    state.ball_y += state.speed_ball_y;
                }
                else if (state.ball_x + this.radius_ball >= this.width - this.thickness_player
                    && state.ball_y + state.world.radius_ball >= state.y && state.ball_y <= state.y + this.width_player) {
                    if (state.speed_ball_x > 0) {
                        state.speed_ball_x *= -1; state.rebound = true;
                    }
                    state.ball_x += state.speed_ball_x;
                }
                else {
                    state.ball_x += state.speed_ball_x;
                    state.ball_y += state.speed_ball_y;
                }

                return state;
            }

            reward(state: QState) {
                if (state.ball_x + this.radius_ball >= this.width - this.thickness_player + 1)
                    return this.reward_loss; //-1000, -1
                else if (state.rebound) {
                    state.rebound = false;
                    return this.reward_win; //100, 0.001
                }
                else
                    return this.reward_living; //10 ma anche 0, 0.0001
            }

            isactionpermitted(state: QState) {
                if (state.y <= 0 && state.action == Action.Down)
                    return false;
                else if (state.y + state.world.width_player >= this.height && state.action == Action.Up)
                    return false;
                else
                    return true;
            }

            isfinalreward(reward: number): boolean {
                if (reward == this.reward_living)
                    return false;
                else
                    return true;
            }
        }

    export class QState implements IQState {
        action: Action;
        world: World;
        vector_mode: VectorMode; //Mode for converting a state to a vector of numbers

        y: number; //Player position as y coordinate
        ball_x: number; //Ball x coordinate
        ball_y: number; //Ball y coordinate
        speed_ball_x: number;
        speed_ball_y: number;
        rebound: boolean = false; //Used for reward when ball is correctly hit
        current_score: number; highest_score: number; //Current and highest score, positive numbers

        constructor(position_of_player: number, x_of_ball: number, y_of_ball: number, action: number, world: World, vector_mode: VectorMode) {
            this.y = position_of_player;
            this.ball_x = x_of_ball;
            this.ball_y = y_of_ball;
            this.action = action;
            this.world = world;
            this.current_score = 0; this.highest_score = 0;
            this.speed_ball_x = world.speed_ball;
            this.speed_ball_y = world.speed_ball;
            this.vector_mode = vector_mode;
        }

        ToVector(): number[] {
            if (this.vector_mode == VectorMode.Difference) {
                var x = new Array<number>(2);
                x[0] = this.y - this.ball_y;
                x[1] = (this.speed_ball_y > 0) ? 0 : 1;
                for (var i = 0; i < 2; i++)
                    x[i] = x[i] / 100;
                return x;
            }
            else if (this.vector_mode == VectorMode.Full) {
                var x = new Array<number>(4);
                x[0] = this.y;
                x[1] = this.ball_y;
                x[2] = this.ball_x;
                if (this.speed_ball_y > 0 && this.speed_ball_x > 0)
                    x[3] = (0);
                else if (this.speed_ball_y > 0 && this.speed_ball_x < 0)
                    x[3] = (1);
                else if (this.speed_ball_y < 0 && this.speed_ball_x > 0)
                    x[3] = (2);
                else if (this.speed_ball_y < 0 && this.speed_ball_x < 0)
                    x[3] = (3);
                for (var i = 0; i < 4; i++)
                    x[i] = x[i] / 100;
                return x;
            }
            else {
                var x = new Array<number>(5);
                x[0] = this.y - this.ball_y;
                x[1] = (this.speed_ball_y > 0) ? 0 : 1;
                x[2] = (this.action == 0) ? 1 : 0;
                x[3] = (this.action == 1) ? 1 : 0;
                x[4] = (this.action == 2) ? 1 : 0;
                return x;
            }
        }

        clone() {
            var newstate = new QState(this.y, this.ball_x, this.ball_y, this.action, this.world, this.vector_mode);
            newstate.y = this.y;
            newstate.action = this.action;
            newstate.ball_x = this.ball_x;
            newstate.ball_y = this.ball_y;
            newstate.speed_ball_x = this.speed_ball_x;
            newstate.speed_ball_y = this.speed_ball_y;
            newstate.current_score = this.current_score; newstate.highest_score = this.highest_score;
            newstate.rebound = this.rebound;
            return newstate;
        }
        
        isfinalstate(): boolean {
            if (this.ball_x >= this.world.width)
                return true;
            else
                return false;
        }
    }
}

module Utilities {
    //Returns an integer in [min, max)
    export function getRandomInt(min, max) {
        return Math.floor(Math.random() * (max - min)) + min;
    }

    export function getRandomOneOrMinusOne() {
        return (Math.random() > 0.5) ? 1 : -1;
    }
}