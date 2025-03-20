import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline


def respond(pipe, question):
	# Define the prompt to generate answers
	prompt = f"{question}";

	# 通过pipe管道生成问题的回答，max_length代表生成答案的最长token数量，num_return_sequences表示生成答案的数量
	# Generate text using the pipeline
	generated_texts = pipe(prompt,
						   max_length=1500,
						   truncation=True,
						   num_return_sequences=1,
						   temperature=0.6,
						   top_p=0.95,
						   do_sample=True
						   );

	# Extract answers from the generated text
	generated_text = generated_texts[0]['generated_text'];

	# The first line is the question repeating.
	answers = generated_text.split("\n")[1:];
	# filter out the None elements
	answer_respond = [answer.strip() for answer in answers if answer.strip()];

	return answer_respond;


if __name__ == '__main__':
	model_path = '/home/kdp/WorkStation/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B';

	tokenizer = AutoTokenizer.from_pretrained(model_path);
	model = AutoModelForCausalLM.from_pretrained(model_path);

	print("Model Load Over!");

	pipe = pipeline("text-generation", model=model, tokenizer=tokenizer);

	while(1):
		question = input("Your talk-> : ");

		if question == "":
			question = "What is the answer of 1 add 9?";

		answer = respond(pipe, question);

		print("Answer:",end="");
		for item in answer:
			print(item);

