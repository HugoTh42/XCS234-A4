from typing import Tuple
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from util import np2torch

LOGSTD_MIN = -10.0
LOGSTD_MAX = 2.0

class ActionSequenceModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        lr: float = 1e-3,
    ):
        """Initialize an action sequence model.

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        segment_len : int
            Action segment length
        lr : float, optional
            Optimizer learning rate, by default 1e-3

        TODO:
        Define self.net to be a neural network with a single hidden layer of size
        hidden_dim that takes as input an observation and outputs the parameters
        to define an action distribution for each time step of the sequence. Use
        ReLU activations, and have the last layer be a linear layer.

        Hint 1: We are predicting an action plan for the entire sequence given
                an observation. What should be the size of the output layer if we
                were to output the actions directly?
        Hint 2: We want the network outputs to be the mean and log standard
                deviation of a distribution from which we sample actions. How can
                we get the output size from the answer to the previous hint?

        Define also self.optimizer to optimize the network parameters. Use a default
        AdamW optimizer with learning rate lr.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.segment_len = segment_len
        #######################################################
        ######### 3-9 lines. ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Return the mean and standard deviation of the action distribution for each observation.

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations

        Returns
        -------
        Tuple[torch.Tensor]
            The means and standard deviations for the actions at future timesteps

        TODO:
        Return mean and standard deviation vectors assuming that self.net predicts
        mean and log std.

        For each observation, your network should have output with dimension
        2 * self.segment_len * self.action_dim. Use the first half of these
        elements to set a mean vector of shape (self.segment_len, self.action_dim)
        in row major order. Use the second half to set a log_std vector of shape
        (self.segment_len, self.action_dim) in row major order. You may want to use
        https://pytorch.org/docs/stable/generated/torch.split.html.

        Hint 1: Apply tanh to the network output mean to force the mean values to
                lie between -1 and 1
        Hint 2: Clamp the log std predictions between LOGSTD_MIN and LOGSTD_MAX
                before converting it to the actual std
        """
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        assert obs.ndim == 2
        batch_size = len(obs)
        net_out = self.net(obs)

        #######################################################
        ######### 3-9 lines. ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################
        return mean, std

    def distribution(self, obs: torch.Tensor) -> D.Distribution:
        """Take in a batch of observations and return a batch of action sequence distributions.

        Parameters
        ----------
        obs : torch.Tensor
            A tensor of observations

        Returns
        -------
        D.Distribution
            The action sequence distributions

        TODO: Given an observation, use self.forward to compute the mean and
        standard deviation of the action sequence distributions, and return the
        corresponding multivariate normal distribution.

        Use distributions.Independent in combination with distributions.Normal
        to create the multivariate normal instead of distributions.MultivariateNormal.
        See https://pytorch.org/docs/stable/distributions.html#independent and
        https://pytorch.org/docs/stable/distributions.html#normal
        """
        #######################################################
        #########   1-5 lines.    ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return an action given an observation

        Parameters
        ----------
        obs : np.ndarray
            Single observation

        Returns
        -------
        np.ndarray
            The selected action

        TODO:
        Predict the full action sequence, and return the first action.

        Hint: Clamp the action values between -1 and 1
        """
        #######################################################
        #########   2-6 lines.    ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################


class SFT(ActionSequenceModel):
    def update(self, obs: torch.Tensor, actions: torch.Tensor):
        """Pre-train a policy given an action sequence for an observation.

        Parameters
        ----------
        obs : torch.Tensor
            The start observation
        actions : torch.Tensor
            A plan of actions for the next timesteps

        TODO:
        Get the underlying action distribution, calculate the log probabilities
        of the given actions, and update the parameters in order to maximize their
        mean.

        Hint: Recall that Pytorch optimizers always try to minimize the loss.
        """
        #######################################################
        #########   4-6 lines.    ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################
        return loss.item()


class DPO(ActionSequenceModel):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        beta: float,
        lr: float = 1e-6,
    ):
        super().__init__(obs_dim, action_dim, hidden_dim, segment_len, lr=lr)
        self.beta = beta

    def update(
        self,
        obs: torch.Tensor,
        actions_w: torch.Tensor,
        actions_l: torch.Tensor,
        ref_policy: nn.Module,
    ):
        """Run one DPO update step

        Parameters
        ----------
        obs : torch.Tensor
            The current observation
        actions_w : torch.Tensor
            The actions of the preferred trajectory
        actions_l : torch.Tensor
            The actions of the other trajectory
        ref_policy : nn.Module
            The reference policy

        TODO:
        Implement the DPO update step.

        Hint 1: When calculating values using the reference policy, use the
                torch.no_grad() context to skip calculating gradients for it and
                achieve better performance
        Hint 2: https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html
        """
        #######################################################
        #########   8-14 lines.   ############
        ### START CODE HERE ###
        ### END CODE HERE ###
        #######################################################
        return loss.item()
