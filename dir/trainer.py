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


    def __init__(self, Policy, model_path="",  lr=0.002, weight_decay=1e-4, momentum = 0, max_grad_norm= torch.inf):

        self.step_model = Policy()
        self.optimizer = torch.optim.SGD(self.step_model.parameters(),
                                    lr=lr,
                                    weight_decay = weight_decay,
                                    momentum = momentum
                                    )

        self.max_grad_norm = max_grad_norm
        if(model_path != "" and Path(model_path).is_file() ):
            checkpoint = torch.load(model_path,weights_only=True)

            #case for the older saves where just saved a model and no optimzer...
            if not "policy" in checkpoint:
                self.step_model.load_state_dict(checkpoint) 
            else:
                self.step_model.load_state_dict(checkpoint["policy"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("loaded_model")

                for g in self.optimizer.param_groups:
                    g['lr'] = lr
                    g['momentum'] = momentum
                    g['weight_decay'] = weight_decay


            print(f"loaded_model from  {model_path}")


        self.value_criterion = nn.MSELoss()
                #observations/state, search_pis, returns are ig the probabilities and values fromMCTS that are being used as targets in trainign
        #ig assuming that are numpy objects
        def train(obs, search_pis, returns):

            #print(obs, search_pis, returns)

            #with torch.autograd.detect_anomaly():  
                self.step_model.train()
                
                obs = torch.from_numpy(obs)
                search_pis = torch.from_numpy(search_pis)
                returns = torch.from_numpy(returns)

                
                self.optimizer.zero_grad()
                logits, policy, value = self.step_model(obs) # the policy isn't actualyl used here... but it's just argmax of logits


                logsoftmax = nn.LogSoftmax(dim=1)
                policy_loss = 5*torch.mean(torch.sum(-search_pis
                                                   * logsoftmax(logits), dim=1)) #using log softmax is like equal to just log compoenet wise of softmax, but i think apprently more numerically stable in implementation? no divison yay?
                                                                                # thinkg the * between tensors is compoentwise multiplciation?
                                                                                #but yeah just taking mean of policy losses
                value_loss = self.value_criterion(value, returns)                    # default for this since not specified when constructed guy is also taking mean of losses

                #print(value,logits,  returns, search_pis)
                #print("pol loss", policy_loss)
                #print("val loss", value_loss)
                loss = policy_loss + value_loss


                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.step_model.parameters(), self.max_grad_norm, error_if_nonfinite=True)

                #print(f"grad_norm: {grad_norm}")

                self.optimizer.step()

                return value_loss.data.numpy(), policy_loss.data.numpy()

        self.train = train
