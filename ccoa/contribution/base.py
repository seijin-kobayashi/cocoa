"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import abc


class Contribution(abc.ABC):
    """
    Base class for the contribution models.
    """

    @abc.abstractmethod
    def reset(self, rng):
        """
        Reset the state.
        """
        pass

    @abc.abstractmethod
    def __call__(self, state, trajectory):
        """
        Compute the contribution of each action in the trajectory.
        """
        pass

    @abc.abstractmethod
    def update(self, rng, state, batch_sampler, offline_batch_sampler, logits_fn):
        """
        Updates the contribution model given a transition.
        """
        pass

    @abc.abstractmethod
    def expected_advantage(self, state, mdp, policy_prob):
        """
        Compute the expected advantage given the ground truth mdp.
        """
        pass
