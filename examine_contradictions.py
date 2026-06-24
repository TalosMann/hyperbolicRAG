import json
import asyncio
import sys
sys.path.insert(0, 'D:/Projects/hyperbolic/Hyper-RAG')
sys.stdout.reconfigure(encoding='utf-8')

async def main():
    from hyperrag.storage import JsonKVStorage
    answers = json.load(open(r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\answers.json', encoding='utf-8'))
    fact_check = json.load(open(r'D:\Projects\hyperbolic\hyperscholar\eval\results\neurology\fact_check.json', encoding='utf-8'))
    fc_by_id = {}
    for c in fact_check['checks']:
        fc_by_id[c['id']] = c
    contradicted_ids = [c['id'] for c in fact_check['checks'] if c['hyperrag']['verdict'] == 'CONTRADICTED']
    answers_by_id = {item['id']: item for item in answers['results']}
    workdir = r'D:\Projects\hyperbolic\hyperscholar\hyperscholar_runtime\hyperrag\neurology'
    gcfg = {'working_dir': workdir, 'addon_params': {}, 'embedding_batch_num': 8}
    chunks = JsonKVStorage(namespace='text_chunks', global_config=gcfg)
    for qid in contradicted_ids:
        item = answers_by_id[qid]
        fc = fc_by_id[qid]
        row = await chunks.get_by_id(item['source_chunk_id'])
        full_text = (row or {}).get('content', '')
        print('=' * 70)
        print('Q' + str(qid))
        print('QUESTION:', item['question'])
        print()
        print('SOURCE PASSAGE:')
        print(full_text[:1000])
        print()
        print('HYPERRAG ANSWER:')
        print(item['hyperrag']['answer'][:500])
        print()
        print('FACT-CHECK EXPLANATION:', fc['hyperrag']['explanation'])
        print()

asyncio.run(main())
