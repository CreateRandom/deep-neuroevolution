#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
source scripts/local_env_setup.sh
python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo $ALGO --num_workers 24