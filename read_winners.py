import json
data = json.load(open(r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\answers.json', encoding='utf-8'))
results = json.load(open(r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\eval_results.json', encoding='utf-8'))
scores_by_id = {}
for q in results['questions']:
    scores_by_id[q['id']] = q.get('scores')

target_ids = [6, 19, 36]
for item in data['results']:
    if item['id'] in target_ids:
        qid = item['id']
        style = item.get('style')
        print('=' * 70)
        print('Q' + str(qid) + ' [' + str(style) + ']')
        print('QUESTION:', item['question'])
        print()
        h = item['hyperrag']
        print('HYPERRAG ok=' + str(h['ok']) + ':')
        print(h['answer'][:600])
        print()
        hi = item['hierarchical']
        nodes = hi['provenance'].get('counts', {}).get('tree_nodes')
        levels = hi['provenance'].get('levels_accessed')
        print('HIERARCHICAL ok=' + str(hi['ok']) + ' nodes=' + str(nodes) + ' levels=' + str(levels) + ':')
        print(hi['answer'][:600])
        print()
        print('SCORES:', scores_by_id.get(qid))
        print()
