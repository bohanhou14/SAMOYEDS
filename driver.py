from engine import Engine
import pickle
from agent import Agent
import torch
import argparse
num_gpus = torch.cuda.device_count()
# print(num_gpus)

# engine = Engine(model_type="mistralai/Mistral-7B-Instruct-v0.1",agents = agents, num_gpus=num_gpus, num_days=3, save_dir="run_cache/debug/")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--num_days", type=int, default=3)
    parser.add_argument("--warmup_days", type=int, default=1)
    parser.add_argument("--profile_path", type=str, default="/home/bhou4/SAMOYEDS/profiles/profiles-num=50.pkl")
    args = parser.parse_args()
    with open(args.profile_path, "rb") as f:
        # a list of dictionaries
        profiles = list(pickle.load(f))
    agents = []
    for p in profiles:
        agent = Agent(profile=p)
        agents.append(agent)

    engine = Engine(profile_str=args.profile_path, model_type=args.model_type, port=args.port, agents = agents, num_gpus=num_gpus, num_days=args.num_days, save_dir="run_cache/debug/")
    engine.run_all_policies(warmup_days=args.warmup_days)

