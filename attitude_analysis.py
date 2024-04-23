import argparse
import pickle
import ast
import matplotlib.pyplot as plt
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    args = parser.parse_args()
    with open(args.data, "rb") as f:
        d = pickle.load(f)
        f.close()
    attitudes = [a[-1]['content'].split("Your answer:")[1] for a in d]
    attitudes = [ast.literal_eval(a) for a in attitudes]

    
    day_0 = [a[0] for a in attitudes]
    day_1 = [a[1] for a in attitudes]
    day_2 = [a[2] for a in attitudes]
    day_0_counter = Counter(day_0)
    day_1_counter = Counter(day_1)
    day_2_counter = Counter(day_2)
    # plot the distribution of attitudes separately, labeling the values of the bars
    # plt.bar(day_0_counter.keys(), day_0_counter.values())
    # plt.title("Attitude Distribution of Day 0")
    # for i, v in enumerate(day_0_counter.values()):
    #     plt.text(i, v, str(v), ha="center")
    # plt.savefig("attitude_distribution_day_0.png")
    # plt.clf()
    # plt.bar(day_1_counter.keys(), day_1_counter.values())
    # plt.title("Attitude Distribution of Day 1")
    # for i, v in enumerate(day_1_counter.values()):
    #     plt.text(i, v, str(v), ha="center")
    # plt.savefig("attitude_distribution_day_1.png")
    # plt.clf()
    # plt.bar(day_2_counter.keys(), day_2_counter.values())
    # plt.title("Attitude Distribution of Day 2")
    # for i, v in enumerate(day_2_counter.values()):
    #     plt.text(i, v, str(v), ha="center")
    # plt.savefig("attitude_distribution_day_2.png")
    # plt.clf()

    # # plot the distribution of attitudes of three days with subplots
    # # label the bar plot with count
    # fig, axs = plt.subplots(3)
    # fig.suptitle("Attitude Distribution of Three Days")
    # axs[0].bar(day_0_counter.keys(), day_0_counter.values())
    # axs[0].set_title("Day 0")
    # axs[1].bar(day_1_counter.keys(), day_1_counter.values())
    # axs[1].set_title("Day 1")
    # axs[2].bar(day_2_counter.keys(), day_2_counter.values())
    # axs[2].set_title("Day 2")
    # plt.savefig("attitude_distribution.png")

    

    convert = 0
    polarize = 0

    for ats in attitudes:
        if (ats[0] == "probably no" or ats[0] == "definitely no") and (ats[2] == "probably yes" or ats[2] == "definitely yes"):
            convert += 1
        if (ats[0] == "probably yes" and ats[2] == "definitely yes"):
            polarize += 1
    print(convert/len(attitudes))
    print(polarize/len(attitudes))
    attitudes_tuple = [(a[0], a[1], a[2]) for a in attitudes]
    print(Counter(attitudes_tuple))
