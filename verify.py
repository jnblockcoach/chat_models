# -*- coding: utf-8 -*-
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'D:\models\qa_model\chat_models\textbook_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total records: {len(data)}')

# Sample 10
print('\n=== Sample 10 Records ===')
for i, rec in enumerate(data[:10]):
    p = rec['paragraph']
    s, e = rec['answer_start'], rec['answer_end']
    ok = (0 <= s < e <= len(p) and p[s:e] == rec['answer'] and 50 <= len(p) <= 300)
    status = 'OK' if ok else 'BAD'
    print(f'[{i+1}] {status}')
    print(f'    P: {p[:70]}...')
    print(f'    Q: {rec["question"][:50]}')
    print(f'    A: {rec["answer"][:50]} (pos: {s}-{e})')
    print()

# verify first 10000
bad = 0
paragraph_issues = 0
for rec in data[:10000]:
    p = rec['paragraph']
    s, e = rec['answer_start'], rec['answer_end']
    if not (0 <= s < e <= len(p) and p[s:e] == rec['answer']):
        bad += 1
    if not (50 <= len(p) <= 300):
        paragraph_issues += 1
print(f'First 10000: invalid answers: {bad}, bad paragraph length: {paragraph_issues}')

# check unique questions
all_questions = [r['question'] for r in data]
print(f'\nAll records: {len(data)}')
print(f'Unique questions: {len(set(all_questions))}')

# Check answer lengths
avg_ans_len = sum(len(r['answer']) for r in data) / len(data)
print(f'Average answer length: {avg_ans_len:.1f} chars')
avg_q_len = sum(len(r['question']) for r in data) / len(data)
print(f'Average question length: {avg_q_len:.1f} chars')
avg_p_len = sum(len(r['paragraph']) for r in data) / len(data)
print(f'Average paragraph length: {avg_p_len:.1f} chars')
