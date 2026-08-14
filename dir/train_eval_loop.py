#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)


import numpy as np
import matplotlib.pyplot as plt
import torch
import pathlib
import json
from importlib import import_module
import time 

from .trainer import Trainer 
from .replay_memory import ReplayMemory
from . import mcts
from .mcts import execute_episode
from .mcts import execute_episode_eval
from .input_reading import get_input


program_start_time = time.time()


args =get_input()

#environment settings
env_module = import_module(f".envs.{args.env_folder}.{args.env_name}", __name__.split(".")[-2])

try:
    problem_specific_stats = import_module(f".envs.{args.env_folder}.additional_statistics",__name__.split(".")[-2]).statistic_functions
except ModuleNotFoundError as e:
    print(f"counldn't find module for problem_specific_stats. error is:{e}")
    problem_specific_stats = {}


print(f"using {args.env_folder}/{args.env_name} as the environment")

assert args.max_step >0


env_module.MAX_STEP = args.max_step
env_module.STEP_PENALTY = args.step_penalty
env_module.HIST_LEN = args.hist_len

try:
    env_module.start_obj_generator = env_module.select_init_method(args.init_method, args.custom_init_list)
except (ModuleNotFoundError, AttributeError) as e:
    print("error occured trying to call environment select_init_method(start_obj_generator, init_method, acustom_list). if your code doesn't need this i guess it's fine")
    print(e)

n_actions = env_module.Env.n_actions
obs_shape = env_module.get_obs_shape()

policy_module=import_module(f".policies.{args.policy}", __name__.split(".")[-2])
Policy =getattr(policy_module, "Policy")
print(f"using policy/model {args.policy}")

#logging and saving state info paths
assert args.dump_path != "" and args.exp_name != "" and args.exp_id !="", "one of dump_path, exp_name, exp_id wasn't specified"

assert args.batch_size < args.memory_size

save_dir= args.dump_path + "/" + args.exp_name + "/" + args.exp_id  + '/'
log_file_path = save_dir + "log_file.txt"
eval_examples_path = save_dir + "eval_"
recent_memory_file_path = save_dir + "recent_mem.npy"
best_mean_memory_file_path = save_dir + "best_mean_mem.npy"
best_min_memory_file_path = save_dir + "best_min_mem.npy"
recent_model_save_state_path = save_dir+ "checkpoint.pth"
best_mean_model_save_state_path = save_dir + "best_mean.pth"
best_min_model_save_state_path = save_dir + "best_min.pth"

easy_acess_vars_file_path = save_dir + "easy_acess_vars.txt"


model_load_path = args.reload_model if args.reload_model!="" else recent_model_save_state_path
mem_load_path = args.reload_mem if args.reload_mem !="" else recent_memory_file_path


#mcts settings
mcts.C_PUCT = args.c_puct
mcts.TEMP_THRESHOLD=args.temp_threshold


#policy settings
assert args.emb_dim % args.num_heads ==0, "pytorch requires the number of heads divide the embedding dimension"

trainer=Trainer( lambda: Policy(
                            num_encoder_layers = args.num_layers, 
                            input_dim= obs_shape[-1], 
                            emb_dim = args.emb_dim,
                            transformer_feedforward_dim = args.transformer_feedforward_dim,
                            encoder_nhead =args.num_heads,
                            num_actions = n_actions
                            ), 
                        lr=args.lr,
                        weight_decay = args.weight_decay, 
                        momentum = args.momentum,
                        max_grad_norm = args.max_grad_norm,
                        model_path=model_load_path 
                )
network = trainer.step_model


#memory stuff
mem = ReplayMemory(args.memory_size,
                   { "ob": np.float32,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []},
                   batch_size = args.batch_size)


try:
    mem.add_all(np.load(mem_load_path,allow_pickle=True).item())
except FileNotFoundError:
    print("no previous memory file (data used to train policy) was found. If you're not loading an existing model this is fine. If you are, it's up to you if you care...")
    



# function to evaluate and report on current agent state
# return best mean and min
def test_agent(num_iterations,current_train_episode, num_min_to_report=1, num_max_to_report = 1):
    #assert not any(torch.isnan(p).any() for p in network.parameters()) , "okay model has nan values. maybe gradient exploding again... ig decrease lr or actually introduce gradient clipping?"

    network.eval()

    stats_list= {} 
    stats_list["obs"]= []
    stats_list["action_list"]= []
    stats_list["reward"]= []
    for stat in problem_specific_stats.keys():
        stats_list[stat] = []


    print("-"*50 +str(current_train_episode)+ "-"*50)


    # do the validation runs and print them and info about them out
    with torch.no_grad():
        for i in range(num_iterations):
            obs, pis, returns, reward, done_state, action_list= execute_episode_eval(network,
                                                                     args.num_simulations,
                                                                    env_module.Env )
            print("observation_list:")
            print(obs)
            print("action_list:")
            print(action_list)
            print("pis:")
            print(pis)
            print("reward:", reward)

            stats_list["obs"].append(obs)
            stats_list["action_list"].append(action_list)
            stats_list["reward"].append(reward)

            for stat_name, stat_fn in problem_specific_stats.items():
                stat_val = stat_fn(obs)
                stats_list[stat_name].append(stat_val)
                print(f"{stat_name}: {stat_val}")
                
            print()
 
    #print basically the same things but order by reward into saved file
    indices_sorted_by_reward =np.argsort(stats_list["reward"])
 
    with open(eval_examples_path + str(current_train_episode)+ ".txt", "w") as file:
        #... our naming isn't great here but idk what do so... leaving for now ig :D 
        for index_num, i in enumerate(indices_sorted_by_reward):
            print(index_num, file = file)

            for key,val in stats_list.items():
                print(key,file=file)
                print(val[i],file=file)
            print(file=file)

    #probbaly should rename stuff but...

    # storing the like data about overall validation run
    statistics= {}
    statistics["avg_reward"]= np.mean(stats_list["reward"]).item()
    statistics["std_reward"]= np.std(stats_list["reward"]).item()
    statistics["min_rewards"]=[ stats_list["reward"][i].item() for i in indices_sorted_by_reward[0:num_min_to_report]]
    statistics["max_rewards"]= [ stats_list["reward"][i].item() for i in indices_sorted_by_reward[-num_max_to_report:None]]
    
    for stat_name in problem_specific_stats.keys():
        avg = np.mean(stats_list[stat_name], axis = 0) 
        avg = avg.item() if np.isscalar(avg) else avg.tolist()
        statistics[stat_name + "_avg(s)"] = avg
 
    
    print()
    for stat,value in statistics.items():
        print(f"{stat}:{value}")

     
    with open(log_file_path, "a") as log_file:
        print(f"eval end time: { (time.time()-program_start_time)/60} minutes", file = log_file)

        print("train_episode:", current_train_episode,file= log_file)
        for stat,value in statistics.items():
            print(f"{stat}:{value}", file = log_file) 
        
        print("__log__", json.dumps(statistics), file = log_file) 
        #this line is so it's easier to read basically for us later. like... if file lengths vary/want to add more statistics, this line will stay one line instead of being multipel so its easiest to just read this line instead of variable number of lnies to read yknow... i'm tired :D

        print(file = log_file)

 
    return statistics["avg_reward"], np.mean(statistics["min_rewards"])


def loop():

    #figure out how many train_episodes we've done
    try:
        with open(easy_acess_vars_file_path , "r") as file:
            start_num_train_episodes = int(file.readline())
            best_mean = float(file.readline())
            best_min = float(file.readline())
            avg_value_losses=file.readline().rstrip("\n").split(",")
            avg_policy_losses=file.readline().rstrip("\n").split(",")

    except FileNotFoundError:
        start_num_train_episodes = 0
        best_mean = -np.inf
        best_min = -np.inf
        avg_value_losses = []
        avg_policy_losses = []

    
    #create folders we're going to store our files in if it doesn't exist
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


    # print the values of the arguments for this program 
    print(json.dumps(vars(args), sort_keys = True, indent = 0)[1:-1])

    with open(log_file_path, "a+") as log_file:
        print("-"*50 + "Start of Settings" + "-"*50,file = log_file)
        print(json.dumps(vars(args),sort_keys=True, indent = 0)[1:-1],file = log_file)
        print( "-"*50 + "Start of Logs" + "-"*50, file = log_file)
    

    
    #actual training now ig

    for i in range(1,args.num_train_episodes+1):
        with torch.no_grad():
            obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                    args.num_simulations,
                                                                    env_module.Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})
 
        # train network and report avg losses
        vl_total= 0
        pl_total =0
        for j in range(args.num_train_step_per_episode):
            batch = mem.get_minibatch()
            vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
            vl_total +=vl
            pl_total +=pl
        avg_value_losses.append( str(vl_total/args.num_train_step_per_episode))
        avg_policy_losses.append( str(pl_total/args.num_train_step_per_episode))

   
        #evaluate agent, then update most recent model and memory
        if i % args.eval_freq== 0: 
            with open(log_file_path, "a+") as file:
                print(f"train end time: { (time.time()-program_start_time)/60} minutes", file = file)

            mean_rew, mean_min_rew = test_agent(args.num_eval_iterations, start_num_train_episodes+i, args.num_min_to_report, args.num_max_to_report)
  
            #update most recent model
            torch.save(network.state_dict(), recent_model_save_state_path)
            print("-"*50 + "model saved" + "-"*50)

            #save most recent memory state

            if mem.count == mem.size:
                mem_save_state = {col_name: np.concatenate((col_content[mem.current:], col_content[0:mem.current])) for col_name,col_content  in mem.columns.items()} 
            else:
                mem_save_state = {col_name: col_content[0:mem.count] for col_name,col_content  in mem.columns.items()}

            np.save(recent_memory_file_path, mem_save_state )

            #update best mean or best min network if needed
            if mean_rew > best_mean: 
                best_mean = mean_rew
                torch.save(network.state_dict(), best_mean_model_save_state_path)

                np.save(best_mean_memory_file_path, mem_save_state)
            if mean_min_rew > best_min:
                best_min = mean_min_rew
                torch.save(network.state_dict(), best_min_model_save_state_path)

                np.save(best_min_memory_file_path, mem_save_state)

            #save numberof train episodes
            with open(easy_acess_vars_file_path, "w") as file:
                print(start_num_train_episodes+i,file = file)
                print(best_mean,file=file)
                print(best_min,file=file)
                print(",".join(avg_value_losses),file = file)
                print(",".join(avg_policy_losses),file = file)

        #periodic save of model and memory state
        if args.save_periodic > 0 and i % args.save_periodic ==0: 
            model_periodic_save_path= save_dir+ f"{start_num_train_episodes+i}.pth"
            torch.save(network.state_dict(), model_periodic_save_path)

            memory_periodic_save_path=save_dir+ f"mem-{start_num_train_episodes+i}.npy"

            np.save(memory_periodic_save_path,mem_save_state)
            


    
