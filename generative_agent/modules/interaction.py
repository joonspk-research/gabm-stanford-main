import math
import sys
import datetime
import random
import string
import re

from numpy import dot
from numpy.linalg import norm

from simulation_engine.settings import * 
from simulation_engine.global_methods import *
from simulation_engine.gpt_structure import *
from simulation_engine.llm_json_parser import *


def _main_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.scratch.self_description}\n==\n"
  agent_desc += f"Private information: {agent.scratch.private_self_description}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=8)
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def _utterance_agent_desc(agent, anchor): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.scratch.self_description}\n==\n"
  agent_desc += f"Speech pattern: {agent.scratch.speech_pattern}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=8)
  nodes = list(retrieved.values())[0]
  for node in nodes:
    agent_desc += f"{node.content}\n"
  return agent_desc


def run_gpt_generate_categorical_resp(
  agent_desc, 
  questions,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Option: {val}\n\n"
    str_questions = str_questions.strip()
    return [agent_desc, str_questions]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_categorical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def categorical_resp(agent, questions): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_categorical_resp(
           agent_desc, questions, "1", LLM_VERS)[0]


def run_gpt_generate_numerical_resp(
  agent_desc, 
  questions, 
  float_resp,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions, float_resp):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Range: {str(val)}\n\n"
    str_questions = str_questions.strip()

    if float_resp: 
      resp_type = "float"
    else: 
      resp_type = "integer"
    return [agent_desc, str_questions, resp_type]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_numerical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions, float_resp) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  if float_resp: 
    output = [float(i) for i in output]
  else: 
    output = [int(i) for i in output]

  return output, [output, prompt, prompt_input, fail_safe]


def numerical_resp(agent, questions, float_resp): 
  anchor = " ".join(list(questions.keys()))
  agent_desc = _main_agent_desc(agent, anchor)
  return run_gpt_generate_numerical_resp(
           agent_desc, questions, float_resp, "1", LLM_VERS)[0]


def run_gpt_generate_utterance(
  agent_desc, 
  str_dialogue,
  context,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, str_dialogue, context):
    return [agent_desc, context, str_dialogue]

  def _func_clean_up(gpt_response, prompt=""): 
    utterance = extract_first_json_dict(gpt_response)["utterance"]
    return utterance

  def _get_fail_safe():
    return None

  prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/utterance_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, str_dialogue, context) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def utterance(agent, curr_dialogue, context): 
  str_dialogue = ""
  for row in curr_dialogue:
    str_dialogue += f"[{row[0]}]: {row[1]}\n"
  str_dialogue += f"[{agent.scratch.get_fullname()}]: [Fill in]\n"

  anchor = str_dialogue
  agent_desc = _utterance_agent_desc(agent, anchor)
  return run_gpt_generate_utterance(
           agent_desc, str_dialogue, context, "1", LLM_VERS)[0]



  





  





  




  





  





