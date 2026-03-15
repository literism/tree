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

with open('/mnt/literism/tree/summary_output/data/oracle_data_model/classification_train.jsonl', 'r') as f:
    lines = f.readlines()

a = b = c = 0
for i, line in enumerate(lines):
    data = json.loads(line)
    output_json = json.loads(data['completion'])
    if i == 2000:
        pass
    if output_json['selected_indices']:
        a += 1
    if output_json['need_new']:
        b += 1
    if output_json['merge_with']:
        c += 1

print(a, b, c, len(lines))