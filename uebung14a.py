# Example for scikit-llm a few-shot prompt using chatgpt
# setup your api keys here: https://platform.openai.com/api-keys.
# DO NOT SHARE THE KEY WITH ANYONE
# This key was revoked after the exercise.
from skllm.config import SKLLMConfig

key = "sk-..."
SKLLMConfig.set_openai_key(key)

X = [
    "I love reading science fiction novels, they transport me to other worlds.",  # example 1 - book - sci-fi
    "A good mystery novel keeps me guessing until the very end.",  # example 2 - book - mystery
    "Historical novels give me a sense of different times and places.",  # example 3 - book - historical
    "I love watching science fiction movies, they transport me to other galaxies.",  # example 4 - movie - sci-fi
    "A good mystery movie keeps me on the edge of my seat.",  # example 5 - movie - mystery
    "Historical movies offer a glimpse into the past.",  # example 6 - movie - historical
]

y = ["books", "books", "books", "movies", "movies", "movies"]
from skllm.models.gpt.classification.few_shot import DynamicFewShotGPTClassifier

query = "I have fallen deeply in love with this sci-fi book; its unique blend of science and fiction has me spellbound."
query2 = (
    "Historical movies offer a glimpse into the past."  # example 6 - movie - historical
)

# DynamicFewShotGPTClassifier samples examples from the training data on the fly.
# that means we don't need to subsample the training data and instead can use the "full" dataset
# for each prediction we get n_examples sampled from the training data

clf = DynamicFewShotGPTClassifier(n_examples=1).fit(X, y)

prompt = clf._get_prompt(query)
print(prompt)
preds = clf.predict([query])
