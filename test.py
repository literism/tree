import json

# with open('/mnt/literism/tree/summary_output/data/iwsft_datasets/iter0/updater_samples.json', 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# a = 0
# b = 0
# c = 0
# for data in dataset:
#     if data['global_reward'] > 0:
#         a += 1
#     if 'Yes' in data['action']:
#         c += 1
#     b += data['global_reward']
#     # print(data['state'])
#     # print(data['action'])
#     # print('-' * 100)
#     # input()
# print(a, b / len(dataset), c, len(dataset))

with open('/Users/literism/Desktop/lab/tree/classification_train.jsonl', 'r') as f:
    lines = f.readlines()

a = b = c = 0
start = 3000
for i, line in enumerate(lines[start:]):
    data = json.loads(line)
    prompt = data['prompt']
    completion = data['completion']
    print(prompt)
    print('=' * 50)
    print('=' * 50)
    print(completion)
    input()

print(a, b, c, len(lines))