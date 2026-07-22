import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import json

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from . import mcts
from .mcts import execute_episode
from .mcts import execute_episode_eval

from .lattice_encoder_policy import Policy
from . import lattice_env as env_module
from .lattice_env import Env, select_init_method



from .input_reading import get_input



args =get_input()


#logging and saving state info paths
assert args.dump_path != "" and args.exp_name != "" and args.exp_id !="", "one of dump_path, exp_name, exp_id wasn't specified"

assert args.batch_size < args.memory_size

save_dir= args.dump_path + "/" + args.exp_name + "/" + args.exp_id  + '/'
log_file_path = save_dir + "log_file.txt"
eval_examples_path = save_dir + "eval_"
memory_file_path = save_dir + "mem.npy"
model_save_state_path = save_dir+ "checkpoint.pth"

num_train_episodes_file_path = save_dir + "num_train_episodes.txt"


model_load_path = args.reload_model if args.reload_model!="" else model_save_state_path


#environment settings
select_init_method(args.init_method, args.custom_init_list)
assert args.max_step >0
env_module.MAX_STEP = args.max_step
n_vectors = 2
vector_dim = 2
n_actions=3

#mcts settings
mcts.C_PUCT = args.c_puct
mcts.TEMP_THRESHOLD=args.temp_threshold


#policy settings
 
encoder_nhead=2 # needs to divide vector_dim...
trainer=trainer=Trainer( lambda: Policy(args.num_layers, vector_dim, encoder_nhead, n_actions), model_path=model_load_path )

obs_shape = [n_vectors, vector_dim]





network = trainer.step_model
#++++++++++++



mem = ReplayMemory(args.memory_size,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []},
                   batch_size = args.batch_size)





# actual train_eval loop stuff is below here ig

def test_agent(num_iterations,current_train_episode):
    network.eval()
    obs_list = []
    action_list_list= []
    reward_list = []
    done_state_list=[]

    for i in range(num_iterations):
        obs, pis, returns, reward, done_state, action_list= execute_episode_eval(network,
                                                                 args.num_simulations,
                                                                 Env )
        print("observation list:")
        print(obs)
        print(action_list)
        print(reward)
        print(f"final state: {done_state}")

        obs_list.append(obs)
        action_list_list.append(action_list)
        reward_list.append(reward)
        done_state_list.append(done_state)

    mean_reward= np.mean(reward_list)          
    std_reward = np.std(reward_list)            
    min_reward = np.min(reward_list)            
    max_reward = np.max(reward_list)            
    #maybe want like worst 10 or smth hoesntly 
    # and to hold on to the observations for those 

    #also would be nice to be able to get performance on specified subsets of training 
    print(f"avg_reward:{mean_reward}") 
    print(f"std_reward:{std_reward}")      
    print(f"min_reward:{min_reward}")
    print(f"max_reward:{max_reward}")

    with open(log_file_path, "a") as log_file:
        print("train_episode:", current_train_episode,file= log_file)
        print(f"avg_reward:{mean_reward}",file= log_file) 
        print(f"std_reward:{std_reward}",file= log_file)      
        print(f"min_reward:{min_reward}",file= log_file)
        print(f"max_reward:{max_reward}",file= log_file)


    with open(eval_examples_path + str(current_train_episode)+ ".txt", "w") as file:
        for i in np.argsort(reward_list):
            print("observation list:", file=file)
            print(obs_list[i], file=file)
            print(action_list_list[i], file=file)
            print(reward_list[i], file=file)
            print(f"final state: {done_state_list[i]}", file=file)

 

def loop():
    try:
        with open(num_train_episodes_file_path, "r") as file:
            start_num_train_episodes = file.readline()

    except FileNotFoundError:
        start_num_train_episodes = 0

    
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    print(json.dumps(vars(args), indent = 0)[1:-1])
    with open(log_file_path, "a+") as log_file:
        print(json.dumps(vars(args), indent = 0)[1:-1],file = log_file)
    

    value_losses = []
    policy_losses = []

    for i in range(1,args.num_train_episodes+1):
        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 args.num_simulations,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})

        batch = mem.get_minibatch()
        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(vl)
        policy_losses.append(pl)


        #update most recent state
        if i % args.eval_freq== 0: 
            test_agent(args.num_eval_iterations, start_num_train_episodes+i)
         
            #update most recent model
            torch.save(network.state_dict(), model_save_state_path)
            print("model saved")

            #save most recent memory state
            np.save(memory_file_path,mem.columns)

            #save numberof train episodes
            with open(num_train_episodes_file_path, "w") as file:
                print(start_num_train_episodes+i,file = file)

        #periodic save of model and memory state
        if args.save_periodic > 0 and i % args.save_periodic ==0: 
            model_periodic_save_path= save_dir+ f"{start_num_train_episodes+i}.pth"
            torch.save(network.state_dict(), periodic_save_path)

            memory_periodic_save_path=save_dir+ f"mem-{start_num_train_episodes+i}.txt"
            with open(memory_periodic-save_path, "w") as file:
                json.dump(mem.columns, file)
 



    #what the command / settings you chose were...
    #logs of how trianing going....?
    #   ig return average + maybe other statistics related to it?
    #    
    # hoenstly maybe the memory guy actually... 
