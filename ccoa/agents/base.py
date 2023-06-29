"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import abc


class Agent(abc.ABC):
    """Base class for an RL agent, which consists in a policy and an update rule."""

    @abc.abstractmethod
    def act(self, rng, agent_state, observation,  eval=False):
        """Return a discrete action sampled from the policy and the action logits"""

    @abc.abstractmethod
    def get_logits(self, agent_state, observation, eval=False):
        """Return the action logits"""

    @abc.abstractmethod
    def reset(self, rng):
        """Return a new agent state"""

    @abc.abstractmethod
    def update(self, rng, agent_state, trajectory, return_contribution):
        """Updates the agent given an episodic trajectory."""
