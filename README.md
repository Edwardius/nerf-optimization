# nerf-optimization
Containerized research environment for NeRF optimization.


## Setup
Run the following before using docker compose to fix ownership while inside a development container:

```bash
$ chmod +x initialize.sh
$ ./initialize.sh
```

This only needs to be done once. Confirm that you're project is initialized by checking the `.env` file. It should contain something like:

```
COMPOSE_PROJECT_NAME=nerf_optimization_e23zhou
FIXUID=1011
FIXGID=1014
```