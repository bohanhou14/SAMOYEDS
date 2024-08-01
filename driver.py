from engines.multi_engine import DataParallelEngine
import argparse

# engine = Engine(model_type="mistralai/Mistral-7B-Instruct-v0.1",agents = agents, num_gpus=num_gpus, num_days=3, save_dir="run_cache/debug/")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--news_path", type=str, default="/home/bhou4/SAMOYEDS/data/news-mixed-k=400.pkl")
    parser.add_argument("--num_days", type=int, default=3)
    parser.add_argument("--warmup_days", type=int, default=1)
    parser.add_argument("--profile_path", type=str, default="/home/bhou4/SAMOYEDS/profiles/profiles-num=50.pkl")
    parser.add_argument("--network_str", type=str, default="/home/bhou4/SAMOYEDS/social_network/social_network-num=50.pkl")
    args = parser.parse_args()

    engine = DataParallelEngine(num_processes=4, news_path=args.news_path, network_str=args.network_str, profile_str=args.profile_path, model_type=args.model_type, num_days=args.num_days, save_dir="run_cache/debug/")
    # print(engine.news)
    engine.run_all_policies(warmup_days=args.warmup_days)

