import re

# Read the config file
with open('config.py', 'r') as f:
    content = f.read()

# Replace all instances of = TRUE with = True
content = re.sub(r'=\s*TRUE\b', '= True', content)

# Replace all instances of = FALSE with = False
content = re.sub(r'=\s*FALSE\b', '= False', content)

# Write the corrected content back
with open('config.py', 'w') as f:
    f.write(content)

print("Fixed all boolean values in config.py")
print("Changed TRUE -> True and FALSE -> False")