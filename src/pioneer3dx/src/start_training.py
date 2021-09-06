#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from gym import wrappers
# ROS packages required

import rospkg
import rospy
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import debugpy
import matplotlib.pyplot as plt

## Debugger
# 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')

# import stable_baselines

# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN


if __name__ == '__main__':
    # env_dict = gym.envs.registration.registry.env_specs.copy()
    # for env in env_dict:
    #     if 'foo' in env:
    #         print("Remove {} from registry".format(env))
    #         del gym.envs.registration.registry.env_specs[env]


    rospy.init_node('pioneer3dx_qlearn',
                    anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/pioneer3dx/task_and_robot_environment_name')
    env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('pioneer3dx')
    # outdir = pkg_path + '/training_results2'
    outdir = '/tmp/gazebo_gym_experiments'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/pioneer3dx/alpha")
    Epsilon = rospy.get_param("/pioneer3dx/epsilon")
    Gamma = rospy.get_param("/pioneer3dx/gamma")
    epsilon_discount = rospy.get_param("/pioneer3dx/epsilon_discount")
    nepisodes = rospy.get_param("/pioneer3dx/nepisodes")
    nsteps = rospy.get_param("/pioneer3dx/nsteps")

    running_step = rospy.get_param("/pioneer3dx/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    all_rewards_mean = []
    number_of_steps = []
    interval_summ = 0
    interval_length = 20

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### WALL START EPISODE=>" + str(x))

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # rospy.logwarn("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logwarn(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            # rospy.logwarn("# state we were=>" + str(state))
            # rospy.logwarn("# action that we took=>" + str(action))
            # rospy.logwarn("# reward that action gave=>" + str(reward))
            # rospy.logwarn("# episode cumulated_reward=>" +
            #               str(cumulated_reward))
            # rospy.logwarn(
            #     "# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                #rospy.logwarn("NOT DONE")
                state = nextState
            else:
                #rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            # rospy.logwarn("############### END Step=>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        interval_summ += cumulated_reward
        if x % interval_length == 0:
            if x == 0:
                all_rewards_mean.append(interval_summ)
            else:
                all_rewards_mean.append(interval_summ / interval_length)
            interval_summ = 0
            number_of_steps.append(x)
            plt.plot(number_of_steps, all_rewards_mean)
            # if x % (interval_length * 5) == 0:
            #     plt.show()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(
        reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
    
    plt.show()

    # model = DQN(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("deepq_cartpole")

    # del model # remove to demonstrate saving and loading

    # model = DQN.load("deepq_cartpole")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    #     env.close()