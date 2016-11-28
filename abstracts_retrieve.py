import os
import string
import re

path = 'test_papers'
for subdirs, dirs, files in os.walk(path):
	for file in files:
		file_path = subdirs + os.path.sep + file
		print(file_path)
		rpFile = open(file_path, 'r')
		text = rpFile.read()
		abstract = re.search('(?s)Abstract(.*?)Introduction', text)
		text_file = open("./abstracts_test/"+file, "w")
		if abstract:
			text_file.write(abstract.group(1))
		text_file.close()	
		rpFile.close()	
			






	
