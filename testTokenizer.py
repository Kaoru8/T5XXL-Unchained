from transformers import AutoTokenizer

rawDir = "PATH TO VANILLA FOLDER HERE"
uncensoredDir = "PATH TO NEW FOLDER HERE"

prompts = [
	"A test prompt goes here",
	"And another one here"
]
rawTokenizer = AutoTokenizer.from_pretrained(rawDir)
patchedTokenizer = AutoTokenizer.from_pretrained(uncensoredDir)
for prompt in prompts:
	tokens = rawTokenizer.tokenize(prompt)
	print("Vanilla tokens:",len(tokens),tokens)
	tokens = patchedTokenizer.tokenize(prompt)
	print("New tokens:",len(tokens),tokens)
	print(80*"-")