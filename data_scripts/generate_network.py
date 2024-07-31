from utils.generate_utils import init_openai_client, init_azure_openai_client, request_azure_generate, request_GPT
import pickle
import argparse
from sandbox.agent import Agent
from tqdm import tqdm

def load_agents(profile_str):
    with open(profile_str, "rb") as f:
        # a list of dictionaries
        profiles = list(pickle.load(f))
    ids = list(range(len(profiles)))
    agents = [Agent(p) for p in profiles]
    for i in range(len(agents)):
        agents[i].id = ids[i]
    return agents

def prompt_ask_follow(profile_str):
    return f"The user has the following profile: {profile_str}. Do you want to follow this user? Return on a scale of 1-4, 4 being most willing to follow and 1 being least willing to follow:"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_str", type=str, default="profiles.pkl")
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    client = init_openai_client(port=args.port)
    agents = load_agents(args.profile_str)
    
    for agent in tqdm(agents, total=len(agents), desc="Total progress"):
        for other_agent in tqdm(agents, total=len(agents), desc="Checking profiles of other agents"):
            if agent == other_agent:
                continue
            prompt = prompt_ask_follow(other_agent.get_profile_str())
            system_prompt = f"Pretend you are someone with this profile: {agent.get_profile_str()}. "
            response = request_GPT(client, prompt, system_prompt=system_prompt, max_tokens=100, model="NousResearch/Hermes-2-Pro-Mistral-7B")
            # print(response)
            if "3" in response or "4" in response:
                if "3" in response:
                    agent.following.append((other_agent.id, 3))
                else:
                    agent.following.append((other_agent.id, 4))
            
        print(f"Following numbers: {len(agent.following)}")
    
    social_network = {}
    for agent in agents:
        social_network[agent.id] = agent.following
    save_path = args.profile_str.replace("profiles", "social_network")
    with open(save_path, "wb") as f:
        pickle.dump(social_network, f)
    



