# Deployment Code of Extreme Parkour on Unitree Go2

This repository provides an **unofficial implementation** for deploying the project [Extreme Parkour with Legged Robots](https://github.com/chengxuxin/extreme-parkour) on the **Unitree Go2** quadrupted robot. The original work was developed for A1 robots and does not provide the deployment code. 

## Key Contributions

- Add detailed comments throughout the training code. Documented the previously unexplained **observation** vector.

- ~~Add camera randomization during training to accout for Go2's movable camera (unlike A1's fixed camera).~~

- Provide train weights and deployment code for Unitree Go2.

## Deployment Instructions

#### Environment Setup
Make sure the environment is properly set up on your Go2 robot, including rclpy, torch and unitree sdk.

#### Hardware Setup
Install the **Intel RealSense D435i** depth camera on the Go2. Verify that the captured images resemble the simulation (can be checked using `rviz`).

#### Deployment Steps
1. Connect to the Go2 robot wirelessly via SSH (wired connection is also ok).
```bash
ssh unitree@<go2_ip_address>
```

2. In the first terminal, start the visual node:
```bash
python3 visual_extreme_parkour.py --logdir traced
```

This script retrieves depth images from the D435i camera and publish them at 100Hz to the appropriate ROS topic.

3. In a second terminal, start the controller node:
```bash
python3 run_extreme_parkour.py --logdir traced --mode parkour --nodryrun
```
This script fuses the depth image and proprioception data. Now the robot is in the sport mode:
- Press **R1** to stand up.
- Press **R2** to lie down.
- Press **X** to balance stand.
- Press **A** to perform a hello gesture (你好).
- Press **L1** to disable the builtin sport service and execute the stand policy.
- After turning off the builtin sport service, press **Y** to start executing the parkour policy.
- When finished, press **L2** to exit the parkour mode and re-enable native motion control.

## Notes and Tips

#### Policy selection:
Modify in `run_extreme_parkour.py`:
```bash
base_model = 'your_base_model.pth'
vision_model = 'your_vision_model.pth'
```

#### Walk Mode:
It is recommended to test the walk mode to verify your model and camera setup:
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk --nodryrun
```
Use ``--mode walk`` or ``--mode parkour`` to switch between **walking** and **parkour** mode, as they were trained as separate tasks in the original work. You can aslo use
```bash
python3 run_extreme_parkour.py --logdir traced --mode walk
```
to perform a more conservative test. This command runs the policy without sending actions to the motors — useful for verifying perception and inference without physical movement.

## Performance 
The Go2 is capable of climbing over obstacles up to **40 cm** in height. Video will be provided soon.

## Acknowledgments
This repository is based on modification of [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour). Special thanks to the original authors for their open-source contribution.

## Contact
I am a beginner in robotics, and warmly welcome feedback and contributions to improve this repository. For questions, suggestions or collaboration, please open an issue or contact me directly.
