import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from queue import PriorityQueue
from queue import Queue
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

import numpy as np

class Client:
    def __init__(self, client_id, inquiry_type, priority_class, arrival_time, service_expected_start_time=0):
        self.client_id = client_id
        self.inquiry_type = inquiry_type  # 'basic' or 'complex'
        self.priority_class = priority_class  # 'normal', 'VIP', etc.
        self.arrival_time = arrival_time
        self.service_expected_start_time = service_expected_start_time
        self.service_start_time = 0
        self.service_expected_end_time = 0
        self.service_end_time = 0
        self.abandonment_time = 0
        self.callback_offer_time = 0
        self.server_id = None
    
    def assign_server(self, server_id):
        self.server_id = server_id

    def copy(self):
            # Return a new Client with the same attributes
            new_client = Client(
                self.client_id,
                self.inquiry_type,
                self.priority_class,
                self.arrival_time,
                self.service_expected_start_time
            )
            # Copy all other attributes
            new_client.service_start_time = self.service_start_time
            new_client.service_expected_end_time = self.service_expected_end_time
            new_client.service_end_time = self.service_end_time
            new_client.abandonment_time = self.abandonment_time
            new_client.callback_offer_time = self.callback_offer_time
            new_client.server_id = self.server_id
            return new_client

class Staff:
    def __init__(self, staff_id):
        self.staff_id = staff_id
        self.available = True
        self.current_client = None
        self.num_clients_served = 0
        self.num_callback_client = 0
        self.num_abandon_client = 0
        self.time_serving_client = 0
        self.staff_queue = Queue()
        self.talking_time = 0
        self.wrap_up_time = 0
        self.last_client_end = 0
        self.last_idle_time = 0
        self.idle_time = 0
        self.num_callback_served = 0

class Staff_Pool:
    def __init__(self, staff_pool_size):
        self.staff_list = [Staff(i) for i in range(staff_pool_size)]
        self.callback_queue = Queue()

    def num_available_staff(self):
        return len([staff for staff in self.staff_list if staff.available])
    
    def assign_client(self, action, client):
        if action < len(self.staff_list) and self.staff_list[action].available:
            self.staff_list[action].available = False
            self.staff_list[action].current_client = client
    
    def available_staff(self):
        return [0 if staff.available else 1 for staff in self.staff_list]

class Event:
    def __init__(self, client_id, time, event_id):
        self.client_id = client_id
        self.time = time
        self.event_id = event_id

class Arrival(Event):
    def __init__(self, client_id, time, enquiry_type, event_id):
        super().__init__(client_id, time, event_id)
        # self.client = client
        self.type = enquiry_type
    
    def __str__(self):
        return f"Arrival T{self.type}"

class Departure(Event):
    def __init__(self, client_id, time, serverid, event_id):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
    
    def __str__(self):
        return f"Departure"

class Abandonment(Event):
    def __init__(self, client_id, time, serverid, event_id):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
        
    def __str__(self):
        return f"Abandonment"

class CallBack(Event):
    def __init__(self, client_id, time, serverid, event_id, accept):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
        self.accept = accept
        
    def __str__(self):
        return f"CallBack"

class WrapUpEnd(Event):
    def __init__(self, client_id, time, serverid, event_id):
        super().__init__(client_id, time, event_id)
        self.serverid = serverid
        
    def __str__(self):
        return f"WrapUpEnd"


    
class CallCentreEnv(gym.Env):
    def __init__(self, parameters,  random_run=True):
        super(CallCentreEnv, self).__init__()
        self.staff_pool_size = len(parameters["service_rate"])
        self.time_until = parameters["time_until"]
        self.arrival_rate = parameters["arrival_rate"]
        self.service_rate = parameters["service_rate"]
        self.abandonment_rate = parameters["abandonment_rate"]
        self.wrap_up_time = parameters["wrap_up_time"]
        self.vip_waiting_time = 300
        self.normal_waiting_time = 600
        
        self.event_queue = PriorityQueue()
        self.staff_pool = Staff_Pool(self.staff_pool_size)
        self.time = 0
        self.client_counter = 0
        self.served_clients_counter = 0
        self.abandonment_counter = 0
        self.event_counter = 0
        self.avg_waiting_time = 0
        self.avg_callback_waiting_time = 0
        self.callback_counter = 0
        self.served_client = []
        self.abandonment_client = []
        self.callback_client = []
        self.happened_event = []
        self.waiting_time_list = []
        self.idle_time_list = []
        self.callback_time_list = []
        self.idle_time = 0

        self.max_staff_queue = 15
        self.max_callback_queue = 30
        self.callback_time = 90  
        self.callback_prob = 0.9 

        self.idle_reward_factor = 10
        # self.service_factor = 1
        # self.arrival_factor = 1
        # self.abandon_factor = 1

        
        
        self.action_space = spaces.Discrete(self.staff_pool_size)
        self.observation_space = spaces.Box(low=0, high=self.max_staff_queue, shape=(self.staff_pool_size * 2 + 3,), dtype=np.float32)

        # Initialize the first arrival event
        init_time = 0
        for i, rate in enumerate(self.arrival_rate):
            self.event_counter += 1
            self.event_queue.put((init_time, Arrival(0, init_time, i, self.event_counter)))
            init_time += 0.0001
        
        if not random_run:
            np.random.seed(1234)
            random.seed(1234)
    
    def reset(self):
        self.event_queue = PriorityQueue()
        self.staff_pool = Staff_Pool(self.staff_pool_size)
        self.time = 0
        self.client_counter = 0
        self.served_clients_counter = 0
        self.abandonment_counter = 0
        self.event_counter = 0
        self.avg_waiting_time = 0
        self.avg_callback_waiting_time = 0
        self.callback_counter = 0
        self.served_client = []
        self.abandonment_client = []
        self.callback_client = []
        self.happened_event = []
        self.waiting_time_list = []
        self.idle_time_list = []
        self.callback_time_list = []
        self.idle_time = 0
         # Reset the latest observation cache
        
        # Initialize the first arrival event
        init_time = 0
        for i, rate in enumerate(self.arrival_rate):
            self.event_counter += 1
            self.event_queue.put((init_time, Arrival(0, init_time, i, self.event_counter)))
            init_time += 0.0001
        
        self._latest_observation = self._get_observation()  

        return self._get_observation() 
    
    def random_with_probability(self):
        return random.random() < self.callback_prob 

    def _get_observation(self):
        obs_size = self.staff_pool_size * 2 + 3
        obs = np.zeros((obs_size,), dtype=np.float32)

        # Fill in staff queue sizes and availability
        for i, staff in enumerate(self.staff_pool.staff_list):
            obs[i*2] = staff.staff_queue.qsize()
            obs[i*2 + 1] = int(staff.available)

        # Callback queue length
        obs[-3] = self.staff_pool.callback_queue.qsize()
        
        # Refined logic for obs[-1]
        callback_queue_not_empty = self.staff_pool.callback_queue.qsize() > 0
        available_staff_with_empty_queue = any(staff.available and staff.staff_queue.qsize() == 0 for staff in self.staff_pool.staff_list)
        
        if callback_queue_not_empty and available_staff_with_empty_queue:
            obs[-1] = 0  # There’s a callback client, staff is available, and no client in their queue
        else:
            obs[-1] = 1 

        # Inquiry type of next event
        if obs[-1] == 1:
            if not self.event_queue.empty():
                _, event = self.event_queue.queue[0]
                obs[-2] = event.type
            else:
                obs[-2] = 0
        
        else:
            obs[-2] = list(self.staff_pool.callback_queue.queue)[0].inquiry_type
                
        return obs
    
    
    def step(self, action):
        reward = 0
        current_state = self._latest_observation
        

        # if self.staff_pool.callback_queue.qsize() > 0 and sum(staff.available for staff in self.staff_pool.staff_list) > 0:
        if current_state[-1] == 0:
            
            self.record_system_state(event_name = "Before Call Back Assignment", action=action, observation=self._latest_observation)
            reward += self._move_callback_client_to_staff(action)
            
            waiting_clients = [client for staff in self.staff_pool.staff_list for client in staff.staff_queue.queue if not staff.staff_queue.empty()]
            num_waiting_clients = len(waiting_clients)

            num_idle_staff = sum(staff.available for staff in self.staff_pool.staff_list)

            idle_time = 0
            if num_idle_staff > 0:
                for staff in self.staff_pool.staff_list:
                    if staff.available:
                        idle_time += self.time - staff.last_idle_time
                        staff.idle_time += self.time - staff.last_idle_time
            
            self.idle_time_list.append(idle_time)
            self.idle_time = idle_time

            reward -= idle_time/self.idle_reward_factor 
            
            num_callback_clients = self.staff_pool.callback_queue.qsize()
            
            self.avg_callback_waiting_time = 0

            if num_callback_clients > 0:
                average_callback_waiting_time = sum(self.time - client.callback_offer_time for client in self.staff_pool.callback_queue.queue)/ num_callback_clients
                self.avg_callback_waiting_time = average_callback_waiting_time
                reward -= average_callback_waiting_time

            self.callback_time_list.append(self.avg_callback_waiting_time)
            
            self.record_system_state(event_name = "After Call Back Assignment", action=action, observation=self._latest_observation)

            reward += self._process_background_events()

            self.record_system_state(event_name = "After Backgroung processing", observation=self._get_observation())
            
            terminated = self.event_queue.empty() and self.staff_pool.callback_queue.qsize() == 0

            self._latest_observation = self._get_observation()  

            return self._latest_observation, reward, terminated, {"action_mask": self.action_masks()}
        
        else:
            # valid_actions = self._available_actions()
            index, event = self.event_queue.get()
            
            selected_staff = self.staff_pool.staff_list[action]
            
            waiting_clients = [client for staff in self.staff_pool.staff_list for client in staff.staff_queue.queue if not staff.staff_queue.empty()]
            num_waiting_clients = len(waiting_clients)
            

            if isinstance(event, Arrival):
                self.time = event.time
                self.client_counter += 1
                
                self.record_system_state(event =event, action=action, observation=self._latest_observation)
                
                new_client = self._client_generator(self.client_counter, event)
                new_client.assign_server(action)
                self.staff_pool.staff_list[action].staff_queue.put(new_client)
                
                # Next Arrival Event
                if self.time < self.time_until:
                    next_arrival_time = self.time + np.random.exponential(self.arrival_rate[event.type])
                    if next_arrival_time < self.time_until: 
                        self.event_counter += 1
                        self.event_queue.put((next_arrival_time, Arrival(new_client.client_id+1, next_arrival_time, event.type, self.event_counter)))
                
                # Abandonment Event
                self.event_counter += 1
                abandonment_time = self.time + np.random.exponential(self.abandonment_rate[event.type])
                self.event_queue.put((abandonment_time, Abandonment(new_client.client_id, abandonment_time, action, self.event_counter)))
                
                # CallBack Event
                self.event_counter += 1
                callback_time = self.time + self.callback_time
                self.event_queue.put((callback_time, CallBack(new_client.client_id, callback_time, action, self.event_counter, self.random_with_probability())))
                
                self.record_system_state(event_name = "After Live Assignment", action=action, observation=self._latest_observation)
                if self.staff_pool.staff_list[action].available and not self.staff_pool.staff_list[action].staff_queue.empty():
                    reward += self._move_client_to_staff(action)

                num_idle_staff = sum(staff.available for staff in self.staff_pool.staff_list)

                self.avg_waiting_time = 0
                if num_waiting_clients > 0:
                    average_waiting_time = sum(self.time - client.arrival_time for client in waiting_clients) / num_waiting_clients
                    self.avg_waiting_time = average_waiting_time
                    reward -= average_waiting_time 
                
                self.waiting_time_list.append(self.avg_waiting_time)

                idle_time = 0
                if num_idle_staff > 0:
                    for staff in self.staff_pool.staff_list:
                        if staff.available:
                            idle_time += self.time - staff.last_idle_time
                            staff.idle_time += self.time - staff.last_idle_time
                
                self.idle_time_list.append(idle_time)
                self.idle_time = idle_time

                reward -= idle_time/self.idle_reward_factor
                
                reward += self._process_background_events()

                self.record_system_state(event_name = "After Backgroung processing", observation=self._get_observation())
                
                terminated = self.event_queue.empty() and self.staff_pool.callback_queue.qsize() == 0

                self._latest_observation = self._get_observation()  

                return self._latest_observation, reward, terminated, {"action_mask": self.action_masks()}



    def _remove_abandonment_by_id(self, target_event_id: int) -> None:
        temp_list = []
        while not self.event_queue.empty():
            priority, event = self.event_queue.get()
            if not (isinstance(event, Abandonment) and event.client_id == target_event_id):
                temp_list.append((priority, event))
        for item in temp_list:
            self.event_queue.put(item)
    
    def _remove_callback_by_id(self, target_event_id: int) -> None:
        temp_list = []
        while not self.event_queue.empty():
            priority, event = self.event_queue.get()
            if not (isinstance(event, CallBack) and event.client_id == target_event_id):
                temp_list.append((priority, event))
        for item in temp_list:
            self.event_queue.put(item)

    def _move_client_to_staff(self, staff_id):
        client = self.staff_pool.staff_list[staff_id].staff_queue.get()
        client.service_start_time = self.time
        
        self._remove_abandonment_by_id(client.client_id)
        self._remove_callback_by_id(client.client_id)
        
        self.staff_pool.staff_list[staff_id].available = False
        self.staff_pool.staff_list[staff_id].current_client = client
        
        # Departure Event
        departure_time = client.service_start_time + np.random.uniform(self.service_rate[staff_id][client.inquiry_type][0], self.service_rate[staff_id][client.inquiry_type][1])
        # departure_time = client.service_start_time + np.random.exponential(self.service_rate[staff_id][client.inquiry_type])
        self.event_counter += 1
        self.event_queue.put((departure_time, Departure(client.client_id, departure_time, staff_id, self.event_counter)))
        
        # Wrap Up Event
        wrapup_time = departure_time + np.random.exponential(self.wrap_up_time[staff_id][client.inquiry_type])
        self.event_counter += 1
        self.event_queue.put((wrapup_time, WrapUpEnd(client.client_id, wrapup_time, staff_id, self.event_counter)))

        if client.service_start_time < client.service_expected_start_time:
            reward = 0
        else:
            reward = -0
        
        return reward

    def _move_callback_client_to_staff(self, staff_id):
        reward = 0
        client = self.staff_pool.callback_queue.get()

        staff = self.staff_pool.staff_list[staff_id]
        staff.num_callback_client += 1

        client.service_start_time = self.time
    
        self.staff_pool.staff_list[staff_id].available = False
        self.staff_pool.staff_list[staff_id].current_client = client
        
        # Departure Event
        departure_time = client.service_start_time + np.random.uniform(self.service_rate[staff_id][client.inquiry_type][0], self.service_rate[staff_id][client.inquiry_type][1])
        # departure_time = client.service_start_time + np.random.exponential(self.service_rate[staff_id][client.inquiry_type])
        self.event_counter += 1
        self.event_queue.put((departure_time, Departure(client.client_id, departure_time, staff_id, self.event_counter)))
        
        # Wrap Up Event
        wrapup_time = departure_time + np.random.exponential(self.wrap_up_time[staff_id][client.inquiry_type])
        self.event_counter += 1
        self.event_queue.put((wrapup_time, WrapUpEnd(client.client_id, wrapup_time, staff_id, self.event_counter)))

        
        return reward

 
    def _process_background_events(self, accumulated_reward=0):

        while not self.event_queue.empty():
            next_time, event = self.event_queue.queue[0]  # Peek at the next event

            if isinstance(event, Arrival):
                # Stop processing when the next event is an Arrival
                break

            # Pop and process background events
            _, event = self.event_queue.get()
            self.time = event.time

            if isinstance(event, Abandonment):
                self.abandonment_counter += 1
                staff = self.staff_pool.staff_list[event.serverid]
                staff.num_abandon_client += 1
                temp_queue = Queue()
                while not staff.staff_queue.empty():
                    client = staff.staff_queue.get()
                    if client.client_id != event.client_id:
                        temp_queue.put(client)
                    else:
                        client.abandonment_time = self.time
                        self.abandonment_client.append(client)
                        self._remove_callback_by_id(client.client_id)
                staff.staff_queue = temp_queue
                self.record_system_state(event=event)
                accumulated_reward -= 125

            elif isinstance(event, CallBack):
                if event.accept:
                    self.callback_counter += 1
                    staff = self.staff_pool.staff_list[event.serverid]
                    temp_queue = Queue()
                    while not staff.staff_queue.empty():
                        client = staff.staff_queue.get()
                        if client.client_id != event.client_id:
                            temp_queue.put(client)
                        else:
                            client.callback_offer_time = self.time
                            self.staff_pool.callback_queue.put(client)
                            self._remove_abandonment_by_id(client.client_id)
                            self.callback_client.append(client.copy())
                    staff.staff_queue = temp_queue
                self.record_system_state(event=event)

            elif isinstance(event, WrapUpEnd):
                staff = self.staff_pool.staff_list[event.serverid]
                staff.available = True
                staff.last_idle_time = self.time
                staff.wrap_up_time += self.time - staff.last_client_end
                self.record_system_state(event=event)
                if staff.staff_queue.qsize() > 0:
                    reward = self._move_client_to_staff(event.serverid)
                    self.record_system_state(event_name="Next Client after Wrap Up")
                    accumulated_reward += reward
                elif staff.available and self.staff_pool.callback_queue.qsize() > 0:
                    break

            elif isinstance(event, Departure):
                staff = self.staff_pool.staff_list[event.serverid]
                self.served_clients_counter += 1
                staff.num_clients_served += 1
                staff.current_client.service_end_time = self.time
                staff.last_client_end = self.time
                staff.talking_time += staff.current_client.service_end_time - staff.current_client.service_start_time
                self.served_client.append(staff.current_client)
                staff.current_client = None
                self.record_system_state(event=event)

        return accumulated_reward



    def _client_generator(self, id, event):
        # Randomly generate client class
        inquiry_type = event.type
        priority_class = random.choice(['normal', 'VIP'])
        expected_waiting_time = self.vip_waiting_time if priority_class == 'VIP' else self.normal_waiting_time
        return Client(id, inquiry_type, priority_class, event.time, event.time + expected_waiting_time)

    def record_system_state(self, event=None, event_name="Manual Snapshot", observation=None, action=None):
        """
        Records a detailed snapshot of the current system state, optionally including an event,
        the observation vector, and the action taken.
        """
        snapshot = {
            "Time": self.time,
            "Event": str(event) if event is not None else event_name,
            "CallBackQueue": [f"Client {client.client_id}: T{client.inquiry_type}" for client in self.staff_pool.callback_queue.queue] if not self.staff_pool.callback_queue.empty() else 'empty',
            "AvailableStaff": self.staff_pool.num_available_staff(),
            "ClientCounter": self.client_counter,
            "ServedClients": self.served_clients_counter,
            "AbandonClients": self.abandonment_counter,
            "CallbackClients": self.callback_counter,
            "IdleTime": self.idle_time,
            "AverageWaitingTime": self.avg_waiting_time,
            "StaffStatus": []
        }

        for staff in self.staff_pool.staff_list:
            if staff.available:
                status = 'Available'
            elif staff.current_client is not None:
                status = f"Serving Client {staff.current_client.client_id}: T{staff.current_client.inquiry_type}"
            else:
                status = "Wrapping Up"
            
            staff_info = {
                "StaffID": staff.staff_id,
                "Status": status,
                "ClientQueue": [f"Client {client.client_id}: T{client.inquiry_type}" for client in staff.staff_queue.queue] if staff.staff_queue.qsize() > 0 else 'empty',
                "NumClientsServed": staff.num_clients_served,
                "NumAbandonClients": staff.num_abandon_client
            }
            snapshot["StaffStatus"].append(staff_info)
        
        # Also record staff queue sizes and the event queue snapshot
        snapshot["StaffQueueSizes"] = [staff.staff_queue.qsize() for staff in self.staff_pool.staff_list]
        snapshot["CallBackQueueSizes"] = self.staff_pool.callback_queue.qsize()
        snapshot["EventQueue"] = [[i, str(j)] for i, j in self.event_queue.queue]
        snapshot["Action Masks"] = self.action_masks().tolist()  # Record action masks as well

        # Optionally record observation and action if provided
        if observation is not None:
            snapshot["Observation"] = observation.tolist() if isinstance(observation, np.ndarray) else observation
        if action is not None:
            snapshot["Action"] = action

        # Save snapshot to happened_event list
        self.happened_event.append(snapshot)



    def _system_statistics(self):
        total_clients = self.client_counter
        total_served = len(self.served_client)
        total_abandonments = len(self.abandonment_client)
        total_callback_offered = self.callback_counter

        # Initialize lists for waiting times
        total_waiting_times = []
        total_callback_waiting_times = []
        total_live_waiting_times = []

        # Initialize per-inquiry-type waiting and service times
        inquiry_type_waiting_times = {i: [] for i in range(len(self.arrival_rate))}
        inquiry_type_service_times = {i: [] for i in range(len(self.arrival_rate))}

        for client in self.served_client:
            waiting_time = client.service_start_time - client.arrival_time
            total_waiting_times.append(waiting_time)

            service_time = client.service_end_time - client.service_start_time
            inquiry_type_service_times[client.inquiry_type].append(service_time)
            inquiry_type_waiting_times[client.inquiry_type].append(waiting_time)

            if client.callback_offer_time > 0:
                callback_waiting = client.service_start_time - client.callback_offer_time
                total_callback_waiting_times.append(callback_waiting)
                live_waiting = client.callback_offer_time - client.arrival_time
                total_live_waiting_times.append(live_waiting)
            else:
                total_live_waiting_times.append(waiting_time)

        total_abandonment_times = [client.abandonment_time - client.arrival_time for client in self.abandonment_client]

        result = {
            "total_clients": total_clients,
            "total_served": total_served,
            "total_abandonments": total_abandonments,
            "abandonment_rate": total_abandonments / total_clients if total_clients > 0 else 0,
            "total_callbacks_offered": total_callback_offered,
            "mean_waiting_time": np.mean(total_waiting_times) if total_waiting_times else 0,
            "mean_live_waiting_time": np.mean(total_live_waiting_times) if total_live_waiting_times else 0,
            "mean_callback_waiting_time": np.mean(total_callback_waiting_times) if total_callback_waiting_times else 0,
            "max_live_waiting_time": np.max(total_live_waiting_times) if total_live_waiting_times else 0,
            "max_callback_waiting_time": np.max(total_callback_waiting_times) if total_callback_waiting_times else 0,
            "mean_abandonment_time": np.mean(total_abandonment_times) if total_abandonment_times else 0
        }

        # Add per-inquiry-type waiting and service time metrics
        for inquiry_type, waiting_list in inquiry_type_waiting_times.items():
            result[f"inquiry_type_{inquiry_type}_mean_waiting_time"] = np.mean(waiting_list) if waiting_list else 0
            # result[f"inquiry_type_{inquiry_type}_max_waiting_time"] = np.max(waiting_list) if waiting_list else 0
            # result[f"inquiry_type_{inquiry_type}_min_waiting_time"] = np.min(waiting_list) if waiting_list else 0

        for inquiry_type, service_list in inquiry_type_service_times.items():
            result[f"inquiry_type_{inquiry_type}_mean_service_time"] = np.mean(service_list) if service_list else 0
            # result[f"inquiry_type_{inquiry_type}_max_service_time"] = np.max(service_list) if service_list else 0
            # result[f"inquiry_type_{inquiry_type}_min_service_time"] = np.min(service_list) if service_list else 0

        # Staff Utilization & Load
        last_time = self.happened_event[-1]["Time"] if self.happened_event else self.time
        clients_served_per_staff = []

        for staff in self.staff_pool.staff_list:
            talking_time = staff.talking_time
            idle_time = staff.idle_time
            clients_served = staff.num_clients_served
            callback_clients_served = staff.num_callback_client
            utilization = talking_time / last_time if last_time > 0 else 0

            clients_served_per_staff.append(clients_served)

            result[f"staff_{staff.staff_id}_talking_time"] = talking_time
            result[f"staff_{staff.staff_id}_idle_time"] = idle_time
            result[f"staff_{staff.staff_id}_utilization"] = utilization
            result[f"staff_{staff.staff_id}_clients_served"] = clients_served
            result[f"staff_{staff.staff_id}_callback_clients_served"] = callback_clients_served
            result[f"staff_{staff.staff_id}_wrap_up_time"] = staff.wrap_up_time
            result[f"staff_{staff.staff_id}_abandonments"] = staff.num_abandon_client

        result["mean_utilization"] = np.mean([result[f"staff_{s.staff_id}_utilization"] for s in self.staff_pool.staff_list])
        result["mean_idle_time"] = np.mean([result[f"staff_{s.staff_id}_idle_time"] for s in self.staff_pool.staff_list])
        result["staff_load_variance"] = np.var(clients_served_per_staff) if clients_served_per_staff else 0

        # Callbacks & Throughput
        callback_clients = list(self.staff_pool.callback_queue.queue)

        result["simulation_time_end"] = last_time

        simulation_duration = last_time
        result["clients_per_hour"] = (total_served / simulation_duration) * 3600 if simulation_duration > 0 else 0
        result["callbacks_per_hour"] = (self.callback_counter / simulation_duration) * 3600 if simulation_duration > 0 else 0

        # Max queue sizes integration
        max_callback_queue_size = max([snapshot["CallBackQueueSizes"] for snapshot in self.happened_event], default=0)
        result["max_callback_queue_size"] = max_callback_queue_size

        for staff_id in range(self.staff_pool_size):
            max_staff_queue_size = max([snapshot["StaffQueueSizes"][staff_id] for snapshot in self.happened_event], default=0)
            result[f"max_staff_{staff_id}_queue_size"] = max_staff_queue_size

        # Metrics from action-based lists
        result["mean_waiting_time_action"] = np.mean(self.waiting_time_list) if self.waiting_time_list else 0
        result["mean_idle_time_action"] = np.mean(self.idle_time_list) if self.idle_time_list else 0
        result["mean_callback_waiting_time_action"] = np.mean(self.callback_time_list) if hasattr(self, 'callback_time_list') and self.callback_time_list else 0

        return result


    def event_list(self):
        return [
            {
                "Time": snapshot.get("Time"),
                "Event": snapshot.get("Event"),
                "CallBackQueue": snapshot.get("CallBackQueue"),
                "EventQueue": snapshot.get("EventQueue"),
                "StaffStatus": snapshot.get("StaffStatus"),
                "AvailableStaff": snapshot.get("AvailableStaff"),
                "ClientCounter": snapshot.get("ClientCounter"),
                "ServedClients": snapshot.get("ServedClients"),
                "AbandonClients": snapshot.get("AbandonClients"),
                "CallbackClients": snapshot.get("CallbackClients"),
                "IdleTime": snapshot.get("IdleTime"),
                "AverageWaitingTime": snapshot.get("AverageWaitingTime"),
                "StaffQueueSizes": snapshot.get("StaffQueueSizes"),
                "Observation": snapshot.get("Observation", "Not Recorded"),
                "Action": snapshot.get("Action", "Not Recorded"),
                "Action Masks": snapshot.get("Action Masks", "Not Recorded"),
                "CallBackQueueSizes": snapshot.get("CallBackQueueSizes", "Not Recorded")
            }
            for snapshot in self.happened_event
        ]


    def action_masks(self):
        # obs = self._get_observation()
        obs = self._latest_observation 
        decision_type = obs[-1]

        # Initialize mask with True (enabled) for all actions
        mask = np.ones((self.action_space.n,), dtype=bool)

        if decision_type == 0:
            # For decision_type 0 (callback client assignment),
            # disable (set False) for staff with full queue or unavailable
            for i, staff in enumerate(self.staff_pool.staff_list):
                if staff.staff_queue.qsize() > 0 or not staff.available:
                    mask[i] = False
        else:
            for i, staff in enumerate(self.staff_pool.staff_list):
                if staff.staff_queue.qsize() >= self.max_staff_queue:
                    mask[i] = False
        return mask



    def render(self, mode='human'):
        print(f"Event Queue Length: {self.event_queue.qsize()}")
        print(f"{[[i,str(j)] for i,j in self.event_queue.queue]}\n")

        print(f"Time: {self.time}")
        print(f"Call Back queue: {[f'Client {client.client_id}: T{client.inquiry_type}' for client in self.staff_pool.callback_queue.queue] if not self.staff_pool.callback_queue.empty() else 'empty'}")
        print(f"Available Staff: {self.staff_pool.num_available_staff()}")
        print(f"Clients Counter: {self.client_counter}")
        print(f"Served Clients: {self.served_clients_counter}")
        print(f"Abandon Clients: {self.abandonment_counter}")
        print(f"Callback Clients: {self.callback_counter}")
        print(f"Idle Time: {self.idle_time}")
        print(f"Average Waiting Time: {self.avg_waiting_time} \n")
            
        print("--- Staff Status ---")
        for staff in self.staff_pool.staff_list:
            if staff.available:
                status = 'Available'
            elif staff.current_client is not None:
                status = f"Serving Client {staff.current_client.client_id}: T{staff.current_client.inquiry_type}"
            else:
                status = "Wrapping Up"
            print(f"Staff {staff.staff_id}: {status}")
            print(f"Staff {staff.staff_id} Client queue: {[f'Client {client.client_id}: T{client.inquiry_type}' for client in staff.staff_queue.queue] if staff.staff_queue.qsize() > 0 else 'empty'}")
            print(f"Number of Clients Served: {staff.num_clients_served}")
            print(f"Number of Abandon Clients: {staff.num_abandon_client}")
            print("\n")
        print("--------------------")




class CallCentreEnvPPO(CallCentreEnv):
    def __init__(self, parameters, random_run=True):
        super().__init__(parameters, random_run)

        # PPO requires a continuous state space: Normalize state to [0,1]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.staff_pool_size * 2 + 3,), dtype=np.float32
        )

    def _get_observation(self):

        obs_size = self.staff_pool_size * 2 + 3
        obs = np.zeros((obs_size,), dtype=np.float32)

        # Fill in staff queue sizes and availability
        for i, staff in enumerate(self.staff_pool.staff_list):
            obs[i*2] = staff.staff_queue.qsize()/ self.max_staff_queue
            obs[i*2 + 1] = int(staff.available)

        # Callback queue length
        obs[-3] = self.staff_pool.callback_queue.qsize()/ self.max_callback_queue
        
        callback_queue_not_empty = self.staff_pool.callback_queue.qsize() > 0
        available_staff_with_empty_queue = any(staff.available and staff.staff_queue.qsize() == 0 for staff in self.staff_pool.staff_list)
        
        if callback_queue_not_empty and available_staff_with_empty_queue:
            obs[-1] = 0  # There’s a callback client, staff is available, and no client in their queue
        else:
            obs[-1] = 1 

        # Inquiry type of next event
        if obs[-1] == 1:
            if not self.event_queue.empty():
                _, event = self.event_queue.queue[0]
                obs[-2] = event.type /len(self.arrival_rate)
            else:
                obs[-2] = 0
        
        else:
            obs[-2] = list(self.staff_pool.callback_queue.queue)[0].inquiry_type /len(self.arrival_rate)

        
        return obs
    
