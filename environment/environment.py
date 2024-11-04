import uuid
import os
import json
import pandas as pd


class Environment:
  def __init__(self, env_type, saved_dir=None):
    self.env_type = env_type
    self.env_id = f'{env_type}_{str(uuid.uuid4())[:15]}'
    self.agent_registry = dict()
    self.responses = None  # Will be different for Survey and Interview

    if saved_dir:
      self._load_saved_env(saved_dir)


  def _load_saved_env(self, saved_dir):
    meta_path = os.path.join(saved_dir, "meta.json")
    agent_registry_path = os.path.join(saved_dir, "agent_registry.json")

    if os.path.exists(meta_path):
      with open(meta_path, 'r') as f:
        meta_info = json.load(f)
        self.env_id = meta_info["env_id"]
      print(f"Loaded meta information from {meta_path}")
    else:
      print(f"Meta file not found at {meta_path}")

    if os.path.exists(agent_registry_path):
      with open(agent_registry_path, 'r') as f:
        self.agent_registry = json.load(f)
      print(f"Loaded agent registry from {agent_registry_path}")
    else:
      print(f"Agent registry file not found at {agent_registry_path}")

    self._load_responses(saved_dir)


  def _load_responses(self, saved_dir):
    # This method will be overridden in child classes
    pass


  def load_agents(self, agent_meta_list):
    new_agent_registry = {f'agent_pid_{str(uuid.uuid4())[:15]}': agent_meta 
                          for agent_meta in agent_meta_list}
    self.agent_registry.update(new_agent_registry)


  def package(self):
    return {
      "packaged_meta": {"env_id": self.env_id},
      "packaged_agents": self.agent_registry,
      "packaged_responses": self._package_responses()
    }


  def _package_responses(self):
    # This method will be overridden in child classes
    pass


  def save(self, save_dir):
    package = self.package()
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "meta.json"), 'w') as json_file:
      json.dump(package["packaged_meta"], json_file, indent=2)

    with open(os.path.join(save_dir, "agent_registry.json"), 'w') as json_file:
      json.dump(package["packaged_agents"], json_file, indent=2)

    self._save_responses(save_dir, package["packaged_responses"])


  def _save_responses(self, save_dir, packaged_responses):
    # This method will be overridden in child classes
    pass

















