python train.py --task neobot_ppo --run_name v1 --num_envs 5000 --max_iterations 2000 --headless #--resume --load_run /home/wx/humanoid/humanoid-gym/logs/neo_ppo/Apr12_14-38-05_v1 --checkpoint 800
################################
#训练一个智能体看效果
#python train.py --task neobot_ppo --run_name v1 --num_envs 1 --max_iterations 2000 
# python train.py --task neobot_ppo --run_name v1 --num_envs 9 --max_iterations 1000 #--headless
# python train.py --task humanoid_ppo --run_name v1 --num_envs 1 --max_iterations 1000
#python train.py --task sigma --run_name v1 --num_envs 1 --max_iterations 1000
# python train.py --task neobot_ppo --run_name v0 --num_envs 999 --max_iterations 1000 