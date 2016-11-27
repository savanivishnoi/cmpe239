import os
import string
import re

path = 'papers_text'
for subdirs, dirs, files in os.walk(path):
	for file in files:
		file_path = subdirs + os.path.sep + file
		rpFile = open(file_path, 'r')
        	text = rpFile.read()
		print file_path		
		abstract = re.search('(?s)Abstract(.*?)Introduction', text)
		if abstract:
			text_file = open("./abstracts/"+file, "w")
			text_file.write(abstract.group(1))
			text_file.close()	
		rpFile.close()	
			






	
