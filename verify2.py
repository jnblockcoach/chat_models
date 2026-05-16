# -*- coding: utf-8 -*-
import json
import os

path = r'D:\models\qa_model\chat_models\textbook_dataset.json'
size = os.path.getsize(path)
print(f'File size: {size / 1024 / 1024:.2f} MB')

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total records: {len(data)}')

# sample 5
print()
for i, rec in enumerate(data[:5]):
    p = rec['paragraph']
    s, e = rec['answer_start'], rec['answer_end']
    ok = (0 <= s < e <= len(p) and p[s:e] == rec['answer'] and 50 <= len(p) <= 300)
    print(f'[{i+1}] {"OK" if ok else "BAD"}')
    print(f'  P: {p[:60]}...')
    print(f'  Q: {rec["question"]}')
    print(f'  A: {rec["answer"]} [{s}:{e}]')
    print()

# verify all
bad = 0
for rec in data:
    p = rec['paragraph']
    s, e = rec['answer_start'], rec['answer_end']
    if not (0 <= s < e <= len(p) and p[s:e] == rec['answer'] and 50 <= len(p) <= 300):
        bad += 1
print(f'Bad records: {bad}/{len(data)}')

# question types
qtypes = {}
for rec in data:
    q = rec['question']
    for t in ['什么', '为什么', '怎么', '多少', '哪里', '什么时候', '谁', '哪一', '请', '如何']:
        if t in q:
            qtypes[t] = qtypes.get(t, 0) + 1
            break
print(f'Question type distribution:')
for t, c in sorted(qtypes.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c}')
