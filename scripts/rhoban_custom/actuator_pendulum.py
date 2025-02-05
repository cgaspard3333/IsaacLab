# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_RHOBAN_CFG  # isort:skip


def design_scene() -> tuple[dict, list]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origin = [0.0, 0.0, 0.0]

    # Articulation
    cartpole_cfg = CARTPOLE_RHOBAN_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # return the scene information
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origin




def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["cartpole"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    
    default_joint_pos = robot.data.default_joint_pos
    default_joint_vel = robot.data.default_joint_vel
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, None)
    mass_body_num = robot.find_bodies("mass")
    mass_mass_body = robot.root_physx_view.get_masses()[0][mass_body_num[0]]
    print(f"[INFO]: Mass at the end of the pendulum: {mass_mass_body[0]} kg")

    
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 10000 == 0:
            # reset counter
            count = 0
            # clear internal buffers
            robot.reset()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, None)
            print("[INFO]: Resetting robot state...")
            mass_body_num = robot.find_bodies("mass")
            mass_mass_body = robot.root_physx_view.get_masses()[0][mass_body_num[0]]
            print(f"[INFO]: Mass: {mass_mass_body}")

        if count % 100 == 0:
            angle = robot.data.joint_pos[0].item()
            print(f"[INFO]: Angle: {angle:.4f} rad")

            
        # Apply random action
        # # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)
        # # -- write data to sim
        # robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.36, 0.0, 0.15], [0.0, 0.0, 0.15])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
