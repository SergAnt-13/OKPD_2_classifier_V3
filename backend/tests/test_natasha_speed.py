from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab
import time

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
vocab = MorphVocab()

text = "молоко 3.2% пастеризованное"
start = time.time()
for _ in range(1000):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(vocab)
    lemmas = [token.lemma for token in doc.tokens]
print(f"1000 раз за {time.time()-start:.2f} сек")