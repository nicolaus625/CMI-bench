from FlagEmbedding import FlagAutoModel
import torch
from torch.nn import functional as F

model = FlagAutoModel.from_finetuned(
    '/import/c4dm-04/siyoul/CMI-bench/pretrained_models/bge-large-en-v1.5',
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
    devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
)

sentence_1 = "I will watch a show tonight"
sentence_2 = "I will show you my watch tonight"
sentence_3 = "I'm going to enjoy a performance this evening"

embeddings = torch.from_numpy(model.encode([sentence_1, sentence_2, sentence_3]))
embedding_1 = embeddings[0].view(1, -1)
embedding_2 = embeddings[1].view(1, -1)
embedding_3 = embeddings[2].view(1, -1)

print(embedding_1.shape)

cos_dist1_2 = F.cosine_similarity(embedding_1, embedding_2).item()
cos_dist1_3 = F.cosine_similarity(embedding_1, embedding_3).item()
print(cos_dist1_2)
print(cos_dist1_3)