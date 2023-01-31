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

## Inside Instant NGP Container
---
```sh
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```

to build and run the thing, do not use the `instant-ngp` executable, instead use

```sh
python3 scripts/run.py path/to/data_images
```

## Inside Kilonerf Container
**BEFORE YOU ENTER** make sure that you mount a your download