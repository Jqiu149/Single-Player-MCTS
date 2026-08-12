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


    def __init__(self, Policy, model_path="",  lr=0.002, weight_decay=1e-4, momentum = 0):

        self.step_model = Policy()
        if(model_path != "" and Path(model_path).is_file() ):
           self.step_model.load_state_dict(torch.load(model_path, weights_only=True)) 
           print("loaded_model")

        value_criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.step_model.parameters(),
                                    lr=lr,
                                    weight_decay = weight_decay,
                                    momentum = momentum
                                    )

        #observations/state, search_pis, returns are ig the probabilities and values fromMCTS that are being used as targets in trainign
        #ig assuming that are numpy objects
        def train(obs, search_pis, returns):

            #print(obs, search_pis, returns)

            #with torch.autograd.detect_anomaly():  
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

                #print(value,logits,  returns, search_pis)
                #print("pol loss", policy_loss)
                #print("val loss", value_loss)
                loss = policy_loss + value_loss


                loss.backward()

                #grad_norm = torch.nn.utils.clip_grad_norm_(self.step_model.parameters(), torch.inf)
                #print(f"grad_norm: {grad_norm}")
                optimizer.step()

                return value_loss.data.numpy(), policy_loss.data.numpy()

        self.train = train
