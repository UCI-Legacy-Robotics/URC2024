#!/bin/bash

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=190 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=90 linkage3:=90 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=90 linkage3:=90 wrist:=90 manipulator_wrist:=90 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=90 linkage3:=90 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=180 linkage1:=0 linkage3:=90 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=90 linkage3:=180 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=-90 linkage3:=0 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=0 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=0 manipulator_wrist:=90 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=0 linkage3:=0 wrist:=90 manipulator_wrist:=90 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=-80 linkage3:=0 wrist:=90 manipulator_wrist:=90 top_claw:=0
sleep 0.25

ros2 launch rover control.launch.py platform:=0 linkage1:=-80 linkage3:=-80 wrist:=90 manipulator_wrist:=90 top_claw:=0

