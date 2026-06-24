import json

q_path = r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\questions.json'
a_path = r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\answers.json'

questions = json.load(open(q_path, encoding='utf-8'))['questions']
answers = json.load(open(a_path, encoding='utf-8'))

q_by_id = {q['id']: q for q in questions}

for item in answers['results']:
    q = q_by_id.get(item['id'])
    if not q:
        continue
    for k, v in q.items():
        if k.startswith('source_') and k not in item:
            item[k] = v

json.dump(answers, open(a_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('backfilled source_* fields into answers.json')
