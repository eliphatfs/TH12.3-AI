# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json

file = r"dql_training.log"
fp = open(file)
json_str = fp.read(-1)
fp.close()

json_obj = json.loads(json_str)

normal_items = ["episode_reward", "mean_absolute_error", "mean_q"]
log_items = ["loss"]
for item in normal_items:
    x = [i for i in range(len(json_obj[item]))]
    y = json_obj[item]
    plt.xlabel(item)
    plt.plot(x, y)
    plt.show()
for item in log_items:
    x = [i for i in range(len(json_obj[item]))]
    y = json_obj[item]
    plt.xlabel(item)
    plt.yscale("log")
    plt.plot(x, y)
    plt.show()
