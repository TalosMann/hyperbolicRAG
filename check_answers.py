import json
data = json.load(open(r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\answers.json', encoding='utf-8'))
for item in data['results']:
    if item['id'] in [37, 38, 41]:
        h = item['hierarchical']
        print('Q' + str(item['id']) + ' [' + str(item.get('style')) + ']')
        print('  ok:', h['ok'])
        print('  answer:', h['answer'][:300])
        print('  provenance counts:', h['provenance'].get('counts'))
        print('  levels_accessed:', h['provenance'].get('levels_accessed'))
        print()
