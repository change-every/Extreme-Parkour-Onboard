import rclpy
from rclpy.node import Node
from unitree_ros2_real import UnitreeRos2Real, get_euler_xyz

import os
import ast
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

from rsl_rl import modules
from rsl_rl.modules import StateHistoryEncoder, RecurrentDepthBackbone, DepthOnlyFCBackbone58x87
import cv2

import sys
import time
import sys
import threading

from sport_api_constants import *

class Go2Node(UnitreeRos2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        self.global_counter = 0
        self.visual_update_interval = 5

        self.actions_sim = torch.from_numpy(np.load('Action_sim_335-11_flat.npy')).to(self.model_device)

        self.sim_ite = 3
 
        self.use_stand_policy = False
        self.use_parkour_policy = False
        self.use_sport_mode = True

    # This warm up is useful in my experiment on Go2
    # The first two iterations are very slow, but the rest is fast
    def warm_up(self):
        for _ in range(2):
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()
            proprio_history = self._get_history_proprio() 
            get_hist_pro_time = time.monotonic()

            depth_image = self._get_depth_image()
            self.depth_latent_yaw = self.depth_encode(depth_image, proprio)

            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)

            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()

            publish_time = time.monotonic()
            print("warm up: ",
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

    def register_models(self, turn_obs, depth_encode, policy):
        self.turn_obs = turn_obs
        self.depth_encode = depth_encode
        self.policy = policy

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):
        if self.use_sport_mode:
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R1):
                self.get_logger().info("In the sport mode, R1 pressed, robot will stand up.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDUP)
            if (self.joy_stick_buffer.keys & self.WirelessButtons.R2):
                self.get_logger().info("In the sport mode, R2 pressed, robot will sit down.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_STANDDOWN)

            if (self.joy_stick_buffer.keys & self.WirelessButtons.X):
                self.get_logger().info("In the sport mode, X pressed, robot will balance stand.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)

            if (self.joy_stick_buffer.keys & self.WirelessButtons.A):
                self.get_logger().info("In the sport mode, A pressed, robot will perform hello gesture.")
                self._sport_mode_change(ROBOT_SPORT_API_ID_HELLO)

            if (self.joy_stick_buffer.keys & self.WirelessButtons.L1):
                self.get_logger().info("Exit the sport mode. Switch to stand policy.")
                self.use_sport_mode = False
                self._sport_state_change(0)
                self.use_stand_policy = True
                self.use_parkour_policy = False
        
        if self.use_stand_policy:
            stand_action = self.get_stand_action()
            self.send_stand_action(stand_action)
        
        if (self.joy_stick_buffer.keys & self.WirelessButtons.Y):
            self.get_logger().info("Y pressed, use the parkour policy")
            self.use_stand_policy = False
            self.use_parkour_policy = True
            self.use_sport_mode = False
            self.global_counter = 0

        if self.use_parkour_policy:
            self.use_stand_policy = False
            self.use_sport_mode = False
            
            start_time = time.monotonic()

            proprio = self.get_proprio()
            get_pro_time = time.monotonic()

            proprio_history = self._get_history_proprio()
            get_hist_pro_time = time.monotonic()

            # print('proprioception: ', proprio)
            # print('history proprioception: ', proprio_history)

            if self.global_counter % self.visual_update_interval == 0:
                depth_image = self._get_depth_image()
                if self.global_counter == 0:
                    self.last_depth_image = depth_image
                self.depth_latent_yaw = self.depth_encode(self.last_depth_image, proprio)
                self.last_depth_image = depth_image
                # print('depth latent: ', self.depth_latent_yaw)
            get_obs_time = time.monotonic()

            obs = self.turn_obs(proprio, self.depth_latent_yaw, proprio_history, self.n_proprio, self.n_depth_latent, self.n_hist_len)
            turn_obs_time = time.monotonic()

            action = self.policy(obs)
            policy_time = time.monotonic()
            # print('action before clip and normalize: ', action)

            # action = self.actions_sim[self.sim_ite, :]
            self.send_action(action)
            print('action: ', action)
            self.sim_ite += 1

            publish_time = time.monotonic()
            print(
                "get proprio time: {:.5f}".format(get_pro_time - start_time),
                "get hist pro time: {:.5f}".format(get_hist_pro_time - get_pro_time),
                "get_depth time: {:.5f}".format(get_obs_time - get_hist_pro_time),
                "get obs time: {:.5f}".format(get_obs_time - start_time),
                "turn_obs_time: {:.5f}".format(turn_obs_time - get_obs_time),
                "policy_time: {:.5f}".format(policy_time - turn_obs_time),
                "publish_time: {:.5f}".format(publish_time - policy_time),
                "total time: {:.5f}".format(publish_time - start_time)
            )

            self.global_counter += 1

        if (self.joy_stick_buffer.keys & self.WirelessButtons.L2):
            self.get_logger().info("L2 pressed, stop using parkour policy, switch to sport mode.")
            self.use_stand_policy = False
            self.use_parkour_policy = False
            self.use_sport_mode = True
            self.reset_obs()
            self._sport_state_change(1)
            self._sport_mode_change(ROBOT_SPORT_API_ID_BALANCESTAND)


@torch.inference_mode()
def main(args):
    rclpy.init()

    assert args.logdir is not None, "Please provide a logdir"
    with open(osp.join(args.logdir, "config.json"), "r") as f:
        config_dict = json.load(f, object_pairs_hook= OrderedDict)
    
    config_dict["control"]["computer_clip_torque"] = True
    
    # duration = config_dict["sim"]["dt"] * config_dict["control"]["decimation"] # different from parkour
    device = "cuda"
    duration = 0.02

    env_node = Go2Node(
        "go2",
        cfg= config_dict,
        model_device= device,
        dryrun= not args.nodryrun,
        mode = args.mode,
    )

    env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir)))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))

    base_model_name = 'base_jit.pt'
    base_model_path = os.path.join(args.logdir, base_model_name)

    vision_model_name = 'vision_weight.pt'
    vision_model_path = os.path.join(args.logdir, vision_model_name)

    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()

    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone

    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])
    depth_encoder.to(device)
    depth_encoder.eval()
    
    def turn_obs(proprio, depth_latent_yaw, proprio_history, n_proprio, n_depth_latent, n_hist_len):
        depth_latent = depth_latent_yaw[:, :-2]
        yaw = depth_latent_yaw[:, -2:] * 1.5
        print('yaw: ', yaw)
        
        proprio[:, 6:8] = yaw

        lin_vel_latent = estimator(proprio)

        activation = nn.ELU()
        priv_latent = hist_encoder(activation, proprio_history.view(-1, n_hist_len, n_proprio))

        
        obs = torch.cat([proprio, depth_latent, lin_vel_latent, priv_latent], dim=-1)

        return obs

    def encode_depth(depth_image, proprio):
        depth_latent_yaw = depth_encoder(depth_image, proprio)
        if torch.isnan(depth_latent_yaw).any():
            print('depth_latent_yaw contains nan and the depth image is: ', depth_image)
        return depth_latent_yaw
    
    def actor_model(obs):
        action = actor(obs)
        return action

    env_node.register_models(turn_obs=turn_obs, depth_encode=encode_depth, policy=actor_model)


    env_node.start_ros_handlers()
    env_node.warm_up()

    if args.loop_mode == "while":
        rclpy.spin_once(env_node, timeout_sec= 0.)
        env_node.get_logger().info("Model and Policy are ready")
        while rclpy.ok():
            main_loop_time = time.monotonic()
            env_node.main_loop()
            rclpy.spin_once(env_node, timeout_sec= 0.)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        env_node.get_logger().info('Model and Policy are ready')
        env_node.start_main_loop_timer(duration)
        rclpy.spin(env_node)

    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--logdir", type= str, default= None, help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--nodryrun", action= "store_true", default= False, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )
    parser.add_argument("--mode", type= str, default= "parkour", choices=["parkour", "walk"])
    args = parser.parse_args()
    
    main(args)
