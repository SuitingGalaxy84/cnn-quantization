import re

label = "Z10.jpg"
label = re.split(r'(\d+)', label)
print(label)