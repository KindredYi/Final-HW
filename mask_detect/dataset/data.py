import os
import re

new_path=os.getcwd()+"\image\\"
print(new_path)
f=open('trainlist_20000.txt','r+')
lines=f.readlines()
print(lines[0])
patten=re.findall(r'(.*?)\\image',lines[0])[0]+"\image\\"

file_data=""
for line in lines:
    line=line.strip('\n').lstrip()
    nline=line.replace(patten,new_path)
    file_data=file_data+nline
    file_data+="\n"
f.close()
f=open('trainlist_20000.txt','r+')
f.write(file_data)
f.close()


