#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)


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
from .encoder_policy import Policy
from .input_reading import get_input


args =get_input()

#probably if statement, based on args thingy ... choose env ig
from . import lattice_env as env_module
from .lattice_env import Env, select_init_method






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


#environment settings
#TODO make it so you can select the environment and ig figure out what other things need to change for those...
assert args.max_step >0
env_module.MAX_STEP = args.max_step
env_module.STEP_PENALTY = args.step_penalty
select_init_method(args.init_method, args.custom_init_list)

n_vectors = 2
vector_dim = 2
n_actions=3
obs_shape = [n_vectors, vector_dim]

#mcts settings
mcts.C_PUCT = args.c_puct
mcts.TEMP_THRESHOLD=args.temp_threshold


#policy settings

assert args.emb_dim % args.num_heads ==0, "pytorch requires the number of heads divide the embedding dimension"

trainer=Trainer( lambda: Policy(
                            num_encoder_layers = args.num_layers, 
                            input_dim=vector_dim, 
                            emb_dim = args.emb_dim,
                            transformer_feedforward_dim = args.transformer_feedforward_dim,
                            encoder_nhead =args.num_heads,
                            num_actions = n_actions
                            ), 
                        lr=args.lr,
                        weight_decay = args.weight_decay, 
                        model_path=model_load_path 
                )
network = trainer.step_model


#memroy stuff
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
    



# evaluate and report on current agent state
# return best mean and min
def test_agent(num_iterations,current_train_episode, num_min_to_report=1, num_max_to_report = 1):
    network.eval()
    obs_list = []
    action_list_list= []
    reward_list = []
    done_state_list=[]

    with torch.no_grad():
        for i in range(num_iterations):
            obs, pis, returns, reward, done_state, action_list= execute_episode_eval(network,
                                                                     args.num_simulations,
                                                                     Env )
            print("observation list:")
            print(obs)
            print("action list:")
            print(action_list)
            print("pis:")
            print(pis)
            print("reward:", reward)

            obs_list.append(obs)
            action_list_list.append(action_list)
            reward_list.append(reward)
            done_state_list.append(done_state)

    indices_sorted_by_reward =np.argsort(reward_list)
    mean_reward= np.mean(reward_list)          
    std_reward = np.std(reward_list)            
    min_rewards= [ reward_list[i] for i in indices_sorted_by_reward[0:num_min_to_report]]
    max_rewards = [ reward_list[i] for i in indices_sorted_by_reward[-num_max_to_report:None]]


    #also would be nice to be able to get performance on specified subsets of training 
    print(f"avg_reward:{mean_reward}") 
    print(f"std_reward:{std_reward}")      
    print(f"min_rewards:{min_rewards}")
    print(f"max_rewards:{max_rewards}")
    print()

    with open(log_file_path, "a") as log_file:
        print("train_episode:", current_train_episode,file= log_file)
        print(f"avg_reward:{mean_reward}",file= log_file) 
        print(f"std_reward:{std_reward}",file= log_file)      
        print(f"min_reward:{min_rewards}",file= log_file)
        print(f"max_reward:{max_rewards}",file= log_file) 
        print(file = log_file)

    with open(eval_examples_path + str(current_train_episode)+ ".txt", "w") as file:
        #... our naming isn't great here but idk what do so... leaving for now ig :D 
        for index_num, i in enumerate(indices_sorted_by_reward):
            print(index_num, file = file)
            print("observation list:", file=file)
            print(obs_list[i], file=file)
            print("action list:", file = file)
            print(action_list_list[i], file=file)
            print(reward_list[i], file=file)
            print(f"final state: {done_state_list[i]}", file=file)


    return mean_reward, np.mean(min_rewards)




def loop():

    #figure out how many train_episodes we've done
    try:
        with open(easy_acess_vars_file_path , "r") as file:
            start_num_train_episodes = int(file.readline())
            best_mean = float(file.readline())
            best_min = float(file.readline())
            avg_value_losses=file.readline().split(",")
            avg_policy_losses=file.readline().split(",")

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
        print("-"*50 + "Start of Settings" + "-"*50)
        print(json.dumps(vars(args),sort_keys=True, indent = 0)[1:-1],file = log_file)
        print( "-"*50 + "Start of Logs" + "-"*50, file = log_file)
    

    #actual training now ig
    for i in range(1,args.num_train_episodes+1):

        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 args.num_simulations,
                                                                 Env)
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


        #update most recent model and memory
        if i % args.eval_freq== 0: 
            mean_rew, mean_min_rew = test_agent(args.num_eval_iterations, start_num_train_episodes+i, args.num_min_to_report, args.num_max_to_report)

#               plt.plot(value_losses, label="value loss")
#               plt.plot(prob_losses, label="action probability loss")
#               plt.legend()
#               plt.show()

         
            #update most recent model
            torch.save(network.state_dict(), recent_model_save_state_path)
            print("-"*50 + "model saved" + "-"*50)

            #save most recent memory state
            np.save(recent_memory_file_path, {col_name: col_content[0:mem.count] for col_name,col_content  in mem.columns.items()})

            #update best mean or best min network if needed
            if mean_rew > best_mean: 
                best_mean = mean_rew
                torch.save(network.state_dict(), best_mean_model_save_state_path)

                np.save(best_mean_memory_file_path, {col_name: col_content[0:mem.count] for col_name,col_content  in mem.columns.items()})
            if mean_min_rew > best_min:
                best_min = mean_min_rew
                torch.save(network.state_dict(), best_min_model_save_state_path)

                np.save(best_min_memory_file_path, {col_name: col_content[0:mem.count] for col_name,col_content  in mem.columns.items()})

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

            np.save(memory_periodic_save_path,mem.columns)




    #what the command / settings you chose were...
    #logs of how trianing going....?
    #   ig return average + maybe other statistics related to it?
    #    
    # hoenstly maybe the memory guy actually... 
