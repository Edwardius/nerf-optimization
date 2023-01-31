#!/bin/bash

FIXUID=$(id -u)
FIXGID=$(id -g)
BASE_PORT=${BASE_PORT:-$(($(id -u)*20))}
GUI_TOOLS_VNC_PORT=${GUI_TOOLS_VNC_PORT:-$((BASE_PORT++))}

> ".env"
echo "COMPOSE_PROJECT_NAME=nerf-optimization-${USER}" >> ".env"
echo "FIXUID=$FIXUID" >> ".env"
echo "FIXGID=$FIXGID" >> ".env"
echo "GUI_TOOLS_VNC_PORT=$GUI_TOOLS_VNC_PORT" >> ".env"
echo "USERNAME=${USER}" >> ".env"