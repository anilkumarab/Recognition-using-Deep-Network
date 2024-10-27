#! /usr/bin/bash

wandb login
python3 /home/arun/PycharmProjects/opencv_projects/project_5/test_wandb.py &

for i in $(seq 1 2)
do
	python3 /home/arun/PycharmProjects/opencv_projects/project_5/test_wandb.py &
done

wait
