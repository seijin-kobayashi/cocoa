"""
Copyright (c) 2023 Alexander Meulemans, Simon Schug, Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from enum import Enum

import itertools
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from .base import Environment, Transition


class ACTIONS(Enum):
    PICK_LEFT = 0
    PICK_RIGHT = 1

@struct.dataclass
class InterleavingStochasticState:
    done: bool
    timestep: int
    open_contexts: jnp.ndarray    # [num_task] you get one of the 2 keys in each open task
    keys: jnp.ndarray             # [num_task]
    current_context: jnp.ndarray  # [1] the task id
    is_answer_room: jnp.ndarray   # [1] bit specifying whether we are in the query room or answer room of the context
    current_objects: jnp.ndarray  # [2] the idx of the object on each side (first: unrewarding). Need to be mapped to actual object
    flipped_objects: jnp.ndarray  # [1] whether the rewarding object is on the right (1) or left(0)


class InterleavingStochastic(Environment):
    def __init__(self, num_contexts, max_parallel_contexts, num_objects, length, die_prob, seed: int) -> None:
        """
         num_contexts: number of "tasks"
         bandwidth: maximum number of context that can be concurrently open
         num_objects: number of rewarded objects. There are as many unrewarded objects
         length: the total env length
        """
        super().__init__()

        self.num_contexts = num_contexts
        self.max_parallel_contexts = max_parallel_contexts
        self.num_objects = num_objects
        self.length = length
        self._reward_values = jnp.concatenate(
            [jnp.linspace(0.2, 1, self.num_contexts), jnp.array([0.])]
        )

        self.zero_to_reward_values = np.where(self._reward_values == 0.0)[0]
        self.die_prob = die_prob
        self.rng = jax.random.PRNGKey(seed)

        # For each context divide objects into rewarded and unrewarded
        @jax.vmap
        def generate_object_splits(rng):
            perm = jax.random.permutation(rng, jnp.arange(self.num_objects * 2))
            return {
                "rewarded": perm[:self.num_objects],
                "unrewarded": perm[self.num_objects:]
            }

        self.object_splits = generate_object_splits(jax.random.split(self.rng, num_contexts))
        
        # Inventory idx are the set of up-to-K-1 hot of num_contexts-1 vector.
        all_inventory_array = [jnp.zeros((1,self.num_contexts-1))]
        for h in range(1, self.max_parallel_contexts):
            perms = itertools.combinations(range(self.num_contexts-1), h)
            inventory_idx = np.array(list(perms)).reshape(-1, h)
            inventory_combin_all_k_hot = jax.vmap(lambda ind: jnp.sum(jax.nn.one_hot(ind, self.num_contexts-1), axis=0))(inventory_idx)
            
            # From the bitmask, create all trit mask
            inventory_combin_all_k_hot_trit = [np.array(list(itertools.product(
                *[list(set([i.item(),2*i.item()])) for i in inventory_combin_k_hot])))
                                               for inventory_combin_k_hot in inventory_combin_all_k_hot]
            all_inventory_array.append(np.concatenate(inventory_combin_all_k_hot_trit))            
        all_inventory_array = jnp.concatenate(all_inventory_array)

        self.all_inventory_array = all_inventory_array
        self.num_inventory_idx = all_inventory_array.shape[0]
        
        self.compute_inventory_idx = lambda v: jax.numpy.argwhere(
            jnp.all(jnp.abs(all_inventory_array - v).astype(int) == 0, axis=1), size=(1), fill_value=-1)[0]

        self.num_state_idx = (self.num_inventory_idx*self.num_contexts*2) + \
                             (self.num_inventory_idx*self.num_contexts*self.num_objects*self.num_objects*2) + \
                              1
        print("Total number of states: ", self.num_state_idx)
        
    def state_to_idx(self, current_context, is_answer_room, current_objects, 
                     flipped_objects, open_contexts, keys, done):
        # all states
        # source (0) + sink (-1)
        # sum of
        #    if is_answer_room == 0:
        #    - flipped_objects: [0,1]
        #    - current_objects: range(num_obj) time range(num_obj)
        #    - current_context: range(num_contexts)
        #    - open_contexts + keys: tritmap of [num_context - 1 ], but with K-1 non 0 terms
        #
        #    if is_answer_room == 1:
        #    - flipped_objects: [0]
        #    - current_objects: [0,0])
        #    - current_context: range(num_contexts)
        #    - open_contexts + keys: tritmap of [num_context - 1 ], but with K-1 non 0 terms, + bit for key or not for current task
        # if T=K, num state: 2*num_obj^2*num_contexts*3^(num_context)
        #                   +num_contexts*3^(num_context)*2

        # The inventory index uniquely identifies the open context and keys, given the current_context
        inventory_array = jnp.delete(open_contexts + keys, current_context, assume_unique_indices=True)
        inventory_idx = self.compute_inventory_idx(inventory_array)
        
        query_room_idx = inventory_idx + \
                (self.num_inventory_idx)*current_context + \
                (self.num_inventory_idx*self.num_contexts)*current_objects[0] + \
                (self.num_inventory_idx*self.num_contexts*self.num_objects)*current_objects[1] + \
                (self.num_inventory_idx*self.num_contexts*self.num_objects*self.num_objects)*flipped_objects 
       
        answer_room_idx = inventory_idx + \
                (self.num_inventory_idx)*current_context + \
                (self.num_inventory_idx*self.num_contexts)*keys[current_context]
        
        not_done_idx = answer_room_idx*is_answer_room + \
               ((self.num_inventory_idx*self.num_contexts*2)+query_room_idx)*~is_answer_room

        # Done state gets the last index
        return not_done_idx*(1-done) + \
              ((self.num_inventory_idx*self.num_contexts*2) + \
               (self.num_inventory_idx*self.num_contexts*self.num_objects*self.num_objects*2))*done
    
    def idx_to_state(self, idx):
        done = (idx == (self.num_inventory_idx*self.num_contexts*2) + \
                     (self.num_inventory_idx*self.num_contexts*self.num_objects*self.num_objects*2))[0]
        
        is_answer_room = (idx <= (self.num_inventory_idx*self.num_contexts*2))[0]
        
        inventory_idx = (idx % self.num_inventory_idx).astype(int)
        idx = idx // self.num_inventory_idx
        current_context = (idx % self.num_contexts).astype(int)
        idx = idx // self.num_contexts
        
        def get_done_state(idx):
            current_key = jnp.zeros((1,)).astype(float)
            current_objects = jnp.zeros((2,)).astype(int)
            flipped_objects = jnp.zeros((1,)).astype(int)
            return current_key, current_objects, flipped_objects
        
        def get_answer_state(idx):
            current_key = (idx % 2).astype(float)
            current_objects = jnp.zeros((2,)).astype(int)
            flipped_objects = jnp.zeros((1,)).astype(int)
            return current_key, current_objects, flipped_objects
                    
        def get_query_state(idx):
            idx = idx - 2
            current_object_0 = idx % self.num_objects
            current_object_1 = (idx // self.num_objects) % self.num_objects
            flipped_objects = ((idx // (self.num_objects*self.num_objects)) % 2).astype(int)
            current_key = jnp.zeros((1,)).astype(float)
            return current_key, jnp.concatenate([current_object_0, current_object_1]).astype(int), flipped_objects
        
        current_key, current_objects, flipped_objects = jax.lax.cond(done,
                                                                    get_done_state,
                                                                    lambda _idx: jax.lax.cond(is_answer_room, 
                                                                                              get_answer_state, 
                                                                                              get_query_state, 
                                                                                              _idx),
                                                                    idx)
        inventory_array = self.all_inventory_array[inventory_idx]
        inventory_array = jnp.insert(inventory_array, current_context, current_key + is_answer_room)
        
        open_contexts = (inventory_array>0).astype(float)
        keys = (inventory_array>1).astype(float)
        
        return current_context, is_answer_room, current_objects, flipped_objects, open_contexts, keys, done
    
    def reset(self, rng):
        rng_object, rng_context, rng_flip = jax.random.split(rng, 3)

        # Set the fist context
        context = jax.random.randint(
            rng_context, shape=(1,), minval=0, maxval=self.num_contexts
        )

        # Sample two objects for each context, one rewarding, one not.
        object_idx = jax.random.randint(
            rng_object, shape=(2,), minval=0, maxval=self.num_objects
        )

        init_state = InterleavingStochasticState(
            done=jnp.array(False),
            timestep=0,
            open_contexts=jnp.zeros(self.num_contexts),
            keys=jnp.zeros(self.num_contexts),
            current_context=context,
            is_answer_room=jnp.array(False),
            current_objects=object_idx,
            flipped_objects=jax.random.randint(rng_flip, shape=(1,), minval=0, maxval=2)
        )
        
        init_transition = Transition(
            observation=self.get_observation(init_state), reward=0.0, done=False, timestep=0
        )

        return init_state, init_transition

    def _get_observation(self, current_context, is_answer_room, current_objects, flipped_objects, open_contexts, keys,
                         done):
        context = jax.nn.one_hot(current_context, self.num_contexts).reshape([-1])
        obj_unrewarded = jax.nn.one_hot(self.object_splits["unrewarded"][current_context, current_objects[0]],
                                        2 * self.num_objects).reshape([-1])
        obj_rewarded = jax.nn.one_hot(self.object_splits["rewarded"][current_context, current_objects[1]],
                                      2 * self.num_objects).reshape([-1])
        objects = (1 - flipped_objects) * jnp.concatenate([obj_rewarded, obj_unrewarded]) + \
                  flipped_objects * jnp.concatenate([obj_unrewarded, obj_rewarded])

        # If answer room, zero out objects
        objects = objects * (~is_answer_room)

        # If done is true, zero out everything
        return jnp.concatenate([context, jnp.expand_dims(is_answer_room, axis=0), objects, open_contexts, keys]) * (~done)

    def get_observation(self, state):
        return self._get_observation(state.current_context,
                                     state.is_answer_room,
                                     state.current_objects,
                                     state.flipped_objects,
                                     state.open_contexts,
                                     state.keys,
                                     state.done)

    def _step(self, rng, state, action):
        rng_die, rng_close, rng_context, rng_object, rng_flip = jax.random.split(rng, 5)

        is_query_room = ~state.is_answer_room
        is_answer_room = state.is_answer_room

        get_key = is_query_room * (action == state.flipped_objects)
        get_reward = is_answer_room * (state.keys[state.current_context])

        reward = (get_reward * self._reward_values[state.current_context])[0]
        open_contexts = state.open_contexts.at[state.current_context].set(is_query_room)
        keys = state.keys.at[state.current_context].set(get_key)

        # Here we contorl the probability of opening new tasks
        close_task = jax.random.uniform(rng_close, ()) < (open_contexts.sum() / self.max_parallel_contexts)
        next_context_idx = jax.random.randint(
            rng_context, shape=(1,), minval=0, maxval=close_task * open_contexts.sum() + \
                                                      (~close_task) * (1 - open_contexts).sum()
        )
        available_contexts = jax.numpy.nonzero(close_task * open_contexts + \
                                               (~close_task) * (1 - open_contexts),
                                               size=self.num_contexts)[0]
        next_context = available_contexts[next_context_idx]
        # Sample two objects for each context, one rewarding, one not.
        object_idx = jax.random.randint(
            rng_object, shape=(2,), minval=0, maxval=self.num_objects
        ) * ~close_task
        flip_objects=jax.random.randint(rng_flip, shape=(1,), minval=0, maxval=2)* ~close_task
        
        # Die or not
        die = jax.random.uniform(rng_die) < self.die_prob

        # Update state
        state = state.replace(
            done=jnp.logical_or(state.timestep >= self.length, die),
            timestep=state.timestep + 1,
            open_contexts=open_contexts,
            keys=keys,

            current_context=next_context,
            is_answer_room=close_task,
            current_objects=object_idx,
            flipped_objects=flip_objects
        )

        transition = Transition(
            observation=self.get_observation(state),
            reward=reward,
            done=state.done,
            timestep=state.timestep,
        )

        return state, transition

    def step(self, rng, state: InterleavingStochasticState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(rng, state: InterleavingStochasticState, action):
            """
            Only update time and give no reward.
            """
            new_timestep = state.timestep + 1
            new_state = state.replace(timestep=new_timestep)
            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=new_timestep,
            )
            return new_state, new_transition

        return jax.lax.cond(
            state.done,
            empty_step,
            self._step,
            rng,
            state,
            action,
        )

    def _step_mdp(self, idx, action):
        current_context, is_answer_room, current_objects, flipped_objects, open_contexts, keys, done = self.idx_to_state(idx)
        is_query_room = ~is_answer_room

        get_key = is_query_room * (action == flipped_objects)
        get_reward = is_answer_room * (keys[current_context])

        reward = (get_reward * self._reward_values[current_context])[0]
        open_contexts = open_contexts.at[current_context].set(is_query_room)
        keys = keys.at[current_context].set(get_key)
        
        ## Compute all possible next states
        # Here we control the probability of opening new tasks
        close_task_pb = (open_contexts.sum() / self.max_parallel_contexts)
        
        all_next_contexts = jnp.arange(self.num_contexts).astype(int)
        all_next_objects = jnp.array(list(itertools.product(np.arange(self.num_objects),np.arange(self.num_objects)))).astype(int)
        all_next_flip = jnp.arange(2).astype(int)
        
        @partial(jax.vmap, in_axes=(0,None,None))
        @partial(jax.vmap, in_axes=(None,0,None))
        @partial(jax.vmap, in_axes=(None,None,0))
        def get_query_transition_pb(next_context, next_objects, next_flip):
            valid = open_contexts[next_context]<1 
            idx = self.state_to_idx(next_context, jnp.array([False]), next_objects, 
                                   next_flip, open_contexts, keys, jnp.array(False))
        
            return jax.nn.one_hot(idx, self.num_state_idx)*valid
        
        @jax.vmap
        def get_answer_transition_pb(next_context):
            valid = open_contexts[next_context]>0 
            idx = self.state_to_idx(next_context, jnp.array([True]), jnp.zeros(2,), 
                                   jnp.zeros(1,), open_contexts, keys, jnp.array(False))
        
            return jax.nn.one_hot(idx, self.num_state_idx)*valid
        
        query_transition_pb = get_query_transition_pb(all_next_contexts, all_next_objects, all_next_flip).sum(axis=[0,1,2,3])
        answer_transition_pb = get_answer_transition_pb(all_next_contexts).sum(axis=[0,1])
        
        transition_probabilities = (1-close_task_pb)*query_transition_pb/query_transition_pb.sum() + \
                                    close_task_pb*answer_transition_pb/answer_transition_pb.sum()

        transition_probabilities = jnp.where(close_task_pb==0, 
                                             query_transition_pb/query_transition_pb.sum(),
                                             transition_probabilities)
        # Die or not
        die_prob = jnp.maximum(self.die_prob, done)
        transition_probabilities = transition_probabilities*(1-die_prob) + jnp.zeros((self.num_state_idx,)).at[-1].set(1)*die_prob
    
        # rewards are deterministic
        reward_probabilities = (self._reward_values == reward) * 1.0  
        
        return transition_probabilities, reward_probabilities


    def step_mdp(self, state: InterleavingStochasticState, action):
        # If environment is already done, don't update the state, but update the time
        def empty_step(state: InterleavingStochasticState, action):
            """
            Only update time and give no reward.
            """
            new_timestep = state.timestep + 1
            new_state = state.replace(timestep=new_timestep)

            reward_probabilities = jnp.zeros(self.reward_values.shape)
            reward_probabilities = reward_probabilities.at[self.zero_to_reward_values].set(1.0)

            new_transition = Transition(
                observation=self.get_observation(state),
                reward=0.0,
                done=state.done,
                timestep=new_timestep,
            )
            return new_state, new_transition, reward_probabilities

        return jax.lax.cond(
            state.done,
            empty_step,
            self._step_mdp,
            state,
            action,
            )

    @property
    def reward_values(self):
        """List of possible rewards returned by the MDP"""
        return self._reward_values

    @property
    def observations(self):
        observation = jax.vmap(lambda idx: self._get_observation(*self.idx_to_state(idx)))(jnp.expand_dims(jnp.arange(self.num_state_idx),-1))
        return observation
    
    @property
    def mdp(self):
        transition, reward = jax.vmap(jax.vmap(self._step_mdp, in_axes=(None,0)), in_axes=(0,None))\
                (jnp.expand_dims(jnp.arange(self.num_state_idx),-1),jnp.arange(self.num_actions))
        return transition, reward
    
    @property
    def init_state(self):
        open_contexts=jnp.zeros(self.num_contexts)
        keys=jnp.zeros(self.num_contexts)
        
        ## Compute all possible next states
        all_next_contexts = jnp.arange(self.num_contexts).astype(int)
        all_next_objects = jnp.array(list(itertools.product(np.arange(self.num_objects),np.arange(self.num_objects)))).astype(int)
        all_next_flip = jnp.arange(2).astype(int)
        
        @partial(jax.vmap, in_axes=(0,None,None))
        @partial(jax.vmap, in_axes=(None,0,None))
        @partial(jax.vmap, in_axes=(None,None,0))
        def get_query_transition_pb(next_context, next_objects, next_flip):
            idx = self.state_to_idx(next_context, jnp.array([False]), next_objects, 
                                   next_flip, open_contexts, keys, jnp.array(False))
        
            return jax.nn.one_hot(idx, self.num_state_idx)
        
        query_transition_pb = get_query_transition_pb(all_next_contexts, all_next_objects, all_next_flip).sum(axis=[0,1,2,3])
        transition_probabilities = query_transition_pb/query_transition_pb.sum() 
                                           
        return transition_probabilities
    
    @property
    def num_actions(self):
        """The number of possible actions"""
        return len(ACTIONS)

    @property
    def observation_shape(self):
        """The shape of the observation array"""
        return (1 + 3*self.num_contexts+4*self.num_objects,)

    def info(self, state):
        """Returns a dictionary of env info useful for logging"""
        log_dict={}
        for i in range(self.num_contexts):
            log_dict[f"frac_key_{i}"] = jnp.where(state.open_contexts[i],state.keys[i],1)
        log_dict["frac_key_all"] = jnp.where(state.open_contexts.sum()==0,1,state.keys.sum()/state.open_contexts.sum())
        return log_dict
