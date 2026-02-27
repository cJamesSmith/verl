sudo python3 -m pip uninstall bytedray -y && sudo python3 -m pip install --force-reinstall "ray[data,train,tune,serve]"
sudo python3 -m pip uninstall byted-wandb -y && sudo python3 -m pip install wandb
# sudo python3 -m pip uninstall byted-wandb -y && sudo python3 -m pip install wandb==0.23.1
sudo python3 -m pip uninstall verl -y
sudo python3 -m pip install protobuf==4.25.8
sudo python3 -m pip install numba==0.63.1
# sudo python3 -m pip install sandbox_fusion
# sudo python3 -m pip install logfire
# sudo python3 -m pip install pydantic-core==2.41.5

# hdfs dfs -rm -r hdfs://harunava/home/byte_malia_gcp_aiic/user/chenxianwei/checkpoints