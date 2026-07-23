import torch
import torch.nn as nn

from pathlib import Path




class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """


    # Policy is a network that outputs LOGITS and value after given a state?
    #learning rate is the one used in optimzer


    def __init__(self, Policy, model_path="",  learning_rate=0.1):

        self.step_model = Policy()
        self.step_model
        if(model_path != "" and Path(model_path).is_file() ):
           self.step_model.load_state_dict(torch.load(model_path, weights_only=True)) 
           print("loaded_model")

        value_criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.step_model.parameters(),
                                    lr=learning_rate)
        # maybe we swap to adam? i think is like generically more liekly to do well ? maybe not?
        # i mean ig adamW for weight decay if want to emulate original thing more

        #observations/state, search_pis, returns are ig the probabilities and values fromMCTS that are being used as targets in trainign
        #ig assuming that are numpy objects
        def train(obs, search_pis, returns):
            self.step_model.train()
            
            obs = torch.from_numpy(obs)
            search_pis = torch.from_numpy(search_pis)
            returns = torch.from_numpy(returns)

            optimizer.zero_grad()
            logits, policy, value = self.step_model(obs) # the policy isn't actualyl used here... but it's just argmax of logits

            logsoftmax = nn.LogSoftmax(dim=1)
            policy_loss = 5*torch.mean(torch.sum(-search_pis
                                               * logsoftmax(logits), dim=1)) #using log softmax is like equal to just log compoenet wise of softmax, but i think apprently more numerically stable in implementation? no divison yay?
                                                                            # thinkg the * between tensors is compoentwise multiplciation?
                                                                            #but yeah just taking mean of policy losses
            value_loss = value_criterion(value, returns)                    # default for this since not specified when constructed guy is also taking mean of losses
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            return value_loss.data.cpu().numpy(), policy_loss.data.cpu().numpy()

        self.train = train
