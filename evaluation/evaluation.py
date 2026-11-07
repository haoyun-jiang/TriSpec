import json
import argparse
import statistics  # 用于计算标准差

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    lines = f.readlines()

print('num of samples:', len(lines))

accept_length = 0
ass_num = 0
base_num = 0
all_num = 0
new_tokens = 0
wall_time = 0
ass_time = 0
base_time = 0
draft_time = 0

cnt = 0

# 新增：存储每条样本的 wall_time 和 speed
per_sample_wall_time = []
per_sample_speed = []

for line in lines:
    data = json.loads(line)
    if "choices" not in data:
        continue
    sample = data['choices'][0]
    if len(sample['turns']) == 0:
        continue

    if "accept_length" in sample:
        if len(sample['accept_length']) > 0:
            accept_length += sum(sample['accept_length']) / len(sample['accept_length']) + 1
        all_num += len(sample['accept_length'])
    if "ass_num" in sample:
        ass_num += sum(sample['ass_num'])
    if "base_num" in sample:
        base_num += sum(sample['base_num'])

    sample_new_tokens = sum(sample['new_tokens'])
    sample_wall_time = sum(sample['wall_time'])

    new_tokens += sample_new_tokens
    wall_time += sample_wall_time

    if "ass_time" in sample:
        ass_time += sum(sample['ass_time'])
    if "base_time" in sample:
        base_time += sum(sample['base_time'])
    if "draft_time" in sample:
        draft_time += sum(sample['draft_time'])

    # 记录每个样本的 wall_time 和 speed
    per_sample_wall_time.append(sample_wall_time)
    if sample_wall_time > 0:
        per_sample_speed.append(sample_new_tokens / sample_wall_time)
    else:
        per_sample_speed.append(0.0)

    cnt += 1

if cnt == 0 or wall_time <= 0:
    raise SystemExit("No valid timed generation records in the input file.")

avg_accept_length = accept_length / cnt
speed = new_tokens / wall_time
ass_time = ass_time / cnt
base_time = base_time / cnt
draft_time = draft_time / cnt

# 计算标准差
wall_time_std = statistics.stdev(per_sample_wall_time) if len(per_sample_wall_time) > 1 else 0.0
speed_std = statistics.stdev(per_sample_speed) if len(per_sample_speed) > 1 else 0.0

# 输出结果
print("acceptance length:", avg_accept_length)
print("ass_num:", ass_num)
print("base_num:", base_num)
print("all_num:", all_num)
print("new_tokens:", new_tokens)
print("wall_time:", wall_time)
print("ass_time:", ass_time)
print("base_time:", base_time)
print("draft_time:", draft_time)
print("speed:", speed)
print("wall_time std:", wall_time_std)
print("speed std:", speed_std)
