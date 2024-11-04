import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from simulation_engine.settings import *
from simulation_engine.global_methods import *
from environment.environment import Environment 
from generative_agent.generative_agent import GenerativeAgent


class Interview(Environment):
  def __init__(self, saved_dir=None):
    super().__init__('interview', saved_dir)
    if not saved_dir: 
      self.responses = {}


  def _load_responses(self, saved_dir):
    responses_path = os.path.join(saved_dir, "responses.json")

    if os.path.exists(responses_path):
      with open(responses_path, 'r') as f:
        self.responses = json.load(f)
      print(f"Loaded responses from {responses_path}")
    else:
      print(f"Responses file not found at {responses_path}")


  def _package_responses(self):
    return self.responses


  def _save_responses(self, save_dir, packaged_responses):
    with open(os.path.join(save_dir, "responses.json"), 'w') as json_file:
        json.dump(packaged_responses, json_file, indent=2)


  def _interview_agent(self, agent_pid, agent_meta, interview_script, context):
    print (f"working on {agent_pid}")
    curr_agent = GenerativeAgent(agent_meta["population"], agent_meta["agent_id"])
    agent_responses = []
    for interview_q, duration in interview_script:
      agent_responses.append(["Interviewer", interview_q])
      agent_response = curr_agent.utterance(agent_responses, context)
      agent_responses.append([curr_agent.scratch.get_fullname(), agent_response])
    return agent_pid, agent_responses

  def interview(self, interview_script, context, num_threads=50):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
      future_to_agent = {executor.submit(self._interview_agent, agent_pid, agent_meta, interview_script, context): agent_pid 
                         for agent_pid, agent_meta in self.agent_registry.items()}
      
      for future in concurrent.futures.as_completed(future_to_agent):
        agent_pid = future_to_agent[future]
        try:
          agent_pid, agent_responses = future.result()
          if agent_pid not in self.responses:
            self.responses[agent_pid] = []

          print (self.responses[agent_pid])
          self.responses[agent_pid] += agent_responses
        except Exception as exc:
          print(f'{agent_pid} generated an exception: {exc}')

    # print (self.responses)
    return self.responses







