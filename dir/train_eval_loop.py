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

num_train_episodes_and_losses_file_path = save_dir + "num_train_episodes_and_losses.txt"


model_load_path = args.reload_model if args.reload_model!="" else model_save_state_path
mem_load_path = args.reload_mem if args.reload_mem !="" else memory_file_path


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
 
encoder_nhead=2 # needs to divide vector_dim...
trainer=trainer=Trainer( lambda: Policy(args.num_layers, vector_dim, encoder_nhead, n_actions), lr=args.lr,weight_decay = args.weight_decay, model_path=model_load_path )



network = trainer.step_model
#++++++++++++


#memroy stuff
mem = ReplayMemory(args.memory_size,
                   { "ob": np.long,
                     "pi": np.float32,
                     "return": np.float32},
                   { "ob":obs_shape,
                     "pi": [n_actions],
                     "return": []},
                   batch_size = args.batch_size)

import numpy as np

try:
    mem.add_all(np.load(mem_load_path,allow_pickle=True).item())
except FileNotFoundError:
    print("no previous memory file (data used to train policy) was found. If you're not loading an existing model this is fine. If you are, it's up to you if you care...")
    



# evaluate and report on current agent state
def test_agent(num_iterations,current_train_episode, num_min_to_report=1, num_max_to_report = 1):
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
        for i in indices_sorted_by_reward:
            print("observation list:", file=file)
            print(obs_list[i], file=file)
            print(action_list_list[i], file=file)
            print(reward_list[i], file=file)
            print(f"final state: {done_state_list[i]}", file=file)




def loop():
    #figure out how many train_episodes we've done
    try:
        with open(num_train_episodes_and_losses_file_path, "r") as file:
            start_num_train_episodes = int(file.readline())
            value_losses=file.readline().split(",")
            prob_losses=file.readline().split(",")

    except FileNotFoundError:
        start_num_train_episodes = 0
        value_losses = []
        prob_losses = []

    
    #create folders we're going to store our files in if it doesn't exist
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


    # print the values of the arguments for this program 
    print(json.dumps(vars(args), indent = 0)[1:-1])
    with open(log_file_path, "a+") as log_file:
        print(json.dumps(vars(args),sort_keys=True, indent = 0)[1:-1],file = log_file)
        print( "-"*50 + "Start of Logs" + "-"*50, file = log_file)
    


    for i in range(1,args.num_train_episodes+1):
        obs, pis, returns, total_reward, done_state = execute_episode(network,
                                                                 args.num_simulations,
                                                                 Env)
        mem.add_all({"ob": obs, "pi": pis, "return": returns})


        batch = mem.get_minibatch()
        vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
        value_losses.append(str(vl))
        prob_losses.append(str(pl))


        #update most recent model and memory
        if i % args.eval_freq== 0: 
            test_agent(args.num_eval_iterations, start_num_train_episodes+i, args.num_min_to_report, args.num_max_to_report)

#               plt.plot(value_losses, label="value loss")
#               plt.plot(prob_losses, label="action probability loss")
#               plt.legend()
#               plt.show()

         
            #update most recent model
            torch.save(network.state_dict(), model_save_state_path)
            print("model saved")

            #save most recent memory state
            np.save(memory_file_path,mem.columns)

            #save numberof train episodes
            with open(num_train_episodes_and_losses_file_path, "w") as file:
                print(start_num_train_episodes+i,file = file)
                print(",".join(value_losses),file = file)
                print(",".join(prob_losses),file = file)


        #periodic save of model and memory state
        if args.save_periodic > 0 and i % args.save_periodic ==0: 
            model_periodic_save_path= save_dir+ f"{start_num_train_episodes+i}.pth"
            torch.save(network.state_dict(), periodic_save_path)

            memory_periodic_save_path=save_dir+ f"mem-{start_num_train_episodes+i}.npy"

            np.save(memory_periodic_save_path,mem.columns)




    #what the command / settings you chose were...
    #logs of how trianing going....?
    #   ig return average + maybe other statistics related to it?
    #    
    # hoenstly maybe the memory guy actually... 
