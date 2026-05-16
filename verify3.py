# -*- coding: utf-8 -*-
import json

with open(r'D:\models\qa_model\chat_models\textbook_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Print 10 records that are NOT from fallback (shorter answers = better quality)
print("=== Records with medium-length answers (6-20 chars) ===")
count = 0
for rec in data:
    alen = len(rec['answer'])
    if 6 <= alen <= 20:
        print(f'P: {rec["paragraph"][:55]}')
        print(f'Q: {rec["question"]}')
        print(f'A: {rec["answer"]} [{rec["answer_start"]}:{rec["answer_end"]}]')
        print()
        count += 1
        if count >= 5:
            break

# Count answer length distribution
alen_dist = {}
for rec in data:
    alen = len(rec['answer'])
    bucket = '<=5' if alen <= 5 else ('6-15' if alen <= 15 else ('16-30' if alen <= 30 else ('31-50' if alen <= 50 else '>50')))
    alen_dist[bucket] = alen_dist.get(bucket, 0) + 1

print('Answer length distribution:')
for b in ['<=5', '6-15', '16-30', '31-50', '>50']:
    print(f'  {b} chars: {alen_dist.get(b, 0)}')

# Check if many records share same paragraph
para_counts = {}
for rec in data:
    para = rec['paragraph'][:30]
    para_counts[para] = para_counts.get(para, 0) + 1

dup_paras = {k:v for k,v in para_counts.items() if v > 3}
print(f'\nParagraphs with >3 duplicates (by first 30 chars): {len(dup_paras)}')
dup_high = sorted(dup_paras.items(), key=lambda x: -x[1])[:5]
for p, c in dup_high:
    print(f'  [{c}x] {p}')
