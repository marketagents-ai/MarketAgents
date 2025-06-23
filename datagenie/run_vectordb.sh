#!/bin/bash

# Run Redis container in detached mode, mapping ports 6379 and 8001
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest