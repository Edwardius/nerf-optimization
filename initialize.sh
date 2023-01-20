#!/bin/bash

FIXUID=$(id -u)
FIXGID=$(id -g)

echo "COMPOSE_PROJECT_NAME=nerf_optimization_${USER}" >> ".env"
echo "FIXUID=$FIXUID" >> ".env"
echo "FIXGID=$FIXGID" >> ".env"