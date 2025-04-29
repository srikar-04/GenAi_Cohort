import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

print(enc.encode("Hello, world!"))  # encoding and decoding are reversable processes
# the output is a list of integers which are "tokens" of these particular models
# if i decode the output "intergers", i will get the original string

print("vocab_size : ", enc.n_vocab) # vocabulary size for the model "gpt-4o"
# each model has it's own vocab size which specifies the no of unique numbers that a model uses for assingnig them to words
