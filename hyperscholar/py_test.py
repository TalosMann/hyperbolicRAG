from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-m3')
v = m.encode(['hello world'])
print('shape:', v.shape)
print('dim:', v.shape[1])
