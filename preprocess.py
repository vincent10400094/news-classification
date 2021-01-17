import json
threshold_count = 3
if __name__ == '__main__':
	count_word = {}
	words_list = []
	punctuation_list = ["，", "。", "、", "：", "；", "？", "！","“","「", "」", "（", "）", "『", "』", "＜", "＞"]
	with open("./icorpus.json") as f:
		datas = json.load(f)
	for data in datas:
		words = data["華語"].split()
		for word in words:
			if word in punctuation_list:
				continue
			elif not count_word.get(word):
				count_word[word] = 1
			else:
				count_word[word] += 1
			if count_word[word] == threshold_count:
				words_list.append(word)
	with open("./words.txt", "w+") as output:
		for word in words_list:
			output.write(word+"\n")
		



