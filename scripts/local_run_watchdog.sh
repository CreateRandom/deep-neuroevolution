PATH=$PATH:$(pwd)
source scripts/local_env_setup.sh
python -m es_distributed.file_watcher $(pwd)