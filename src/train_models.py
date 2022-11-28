import os
import numpy as np
import subprocess
from subprocess import Popen, PIPE

num_frames = 300
num_trials = 10

scores_list = []

controller_freq_options = [0.0, 0.5, 1.0]
LR_options = [0.1]
use_Q_options = ["yes", "no", "no_without_efferent"]

explore_amount_options = [0.5, 1.0, 2.0, 4.0, 8.0]

fp_name_options = ["ff_trained", "ff_trained_short", "ff_trained_random"]#, "ff_trained_biased"]

test_freq = 3

A_coef_options = [0.5**5, 0.5**4,  0.5**3, 0.5**2, 0.5**1]

task_options = ["TwoJoint", "Maze"]
combo_options = ["sample", "mean"]

count = 0
for trial in [0, 1, 2, 3, 4]:
    for learn_A_coef in [0]:
        
        for task in ["TwoJoint", "Maze"]:
            for combo in combo_options:
    
                print('trial', trial)

                for explore_amount in explore_amount_options:

                    print('explore amount', explore_amount)
                    fp_idx = -1

                    processes = []
                    for fp_name in fp_name_options:
                        print('fp name', fp_name)

                        fp_idx += 1
                        
                        for controller_freq in [0.0, 0.5, 1.0]:
                            print('controller freq', controller_freq)
                            
                            for A_coef in A_coef_options:

                                for LR in LR_options:
                                    for use_Q in use_Q_options:

                                        count += 1
                                        my_env = os.environ
                                        my_env["CUDA_VISIBLE_DEVICES"] = str(count % 4)
                                        p = Popen(["python3", "model.py", "--trial", str(trial), "--explore_amount", str(explore_amount), "--fp_name", str(fp_name), "--controller_freq", str(controller_freq), "--LR", str(LR), "--use_Q", str(use_Q), "--num_frames", str(num_frames), "--A_coef", str(A_coef), "--learn_A_coef", str(learn_A_coef), "--task", str(task), "--combo_type", str(combo)],
                                                stdout=PIPE, stderr=PIPE, env=my_env)
                                        processes.append(p)

                    for p in processes:
                        p.wait()
                        output, error = p.communicate()
                        if p.returncode != 0: 
                           print("failed %d %s %s" % (p.returncode, output, error))
