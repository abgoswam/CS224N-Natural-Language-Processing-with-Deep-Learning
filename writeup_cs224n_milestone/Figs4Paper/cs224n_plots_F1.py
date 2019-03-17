import numpy as np
import matplotlib.pyplot as plt

baseline_file_name = r'H:\_hackerreborn\private-repo\save\train\baseline-01-baseline-bidaf\log.txt'
model_file_name = r'H:\_hackerreborn\private-repo\save\train\baseline-10-dropout-point2-unfrozen\log.txt'

# baseline

baseline_file_contents = None
with open(baseline_file_name) as log_file:
    baseline_file_contents = [line.rstrip('\n') for line in log_file]

print(baseline_file_contents)
baseline_step_contents = [line for line in baseline_file_contents if "Evaluating at" in line]
baseline_stats_contents = [line for line in baseline_file_contents if "Dev NLL" in line]

baseline_step = []
for line in baseline_step_contents:
    line_tok = line.split()
    baseline_step.append(int(line_tok[5].strip('.')))

baseline_NLL = []
baseline_F1 = []
baseline_EM = []
baseline_AvNA = []

for line in baseline_stats_contents:
    line_tok = line.split()
    baseline_NLL.append(float(line_tok[4].strip(',')))
    baseline_F1.append(float(line_tok[6].strip(',')))
    baseline_EM.append(float(line_tok[8].strip(',')))
    baseline_AvNA.append(float(line_tok[10].strip(',')))

baseline_step = np.array(baseline_step) / 1000000
baseline_NLL = np.array(baseline_NLL)

# our model
model_file_contents = None
with open(model_file_name) as log_file:
    model_file_contents = [line.rstrip('\n') for line in log_file]

print(model_file_contents)
model_step_contents = [line for line in model_file_contents if "Evaluating at" in line]
model_stats_contents = [line for line in model_file_contents if "Dev NLL" in line]

model_step = []
for line in model_step_contents:
    line_tok = line.split()
    model_step.append(int(line_tok[5].strip('.')))

model_NLL = []
model_F1 = []
model_EM = []
model_AvNA = []

for line in model_stats_contents:
    line_tok = line.split()
    model_NLL.append(float(line_tok[4].strip(',')))
    model_F1.append(float(line_tok[6].strip(',')))
    model_EM.append(float(line_tok[8].strip(',')))
    model_AvNA.append(float(line_tok[10].strip(',')))

model_step = np.array(model_step) / 1000000
model_LL = np.array(model_NLL)

#
fig, ax = plt.subplots()
ax.plot(baseline_step, baseline_NLL, 'bo-', label='BiDAF (baseline)')
ax.plot(baseline_step, model_NLL, 'ro-', label='Our Model')
ax.set(xlabel='Step Count (in million)', ylabel='Negative Log Likelihood (NLL) Loss', title='NLL Loss')
ax.grid(True)
plt.legend(loc=0)
fig.savefig("NLL.png")
plt.show()
#

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(baseline_step, baseline_F1, 'bo-', label='BiDAF (baseline)')
ax1.plot(baseline_step, model_F1, 'ro-', label='Our Model')
ax1.set(ylabel='F1 Score')
ax1.grid(True)
ax1.legend(loc=0)

ax2.plot(baseline_step, baseline_EM, 'bo-', label='BiDAF (baseline)')
ax2.plot(baseline_step, model_EM, 'ro-', label='Our Model')
ax2.set(xlabel='Step Count (in million)', ylabel='Exact Match (EM) Score')
ax2.grid(True)
ax2.legend(loc=0)

fig.savefig("F1_EM.png")
plt.show()
