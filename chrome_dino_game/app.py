import os
import re
import json

# Directory to search for Python files
project_dir = '.'

# Regex patterns for 'import' and 'from' statements
import_pattern = re.compile(r'^\s*import (\S+)')
from_pattern = re.compile(r'^\s*from (\S+)')

# Set to store unique libraries
libraries = set()

# Function to process .ipynb files
def process_ipynb(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        # Look through all the cells in the notebook
        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'code':
                for line in cell['source']:
                    # Check for import and from statements
                    match = import_pattern.match(line) or from_pattern.match(line)
                    if match:
                        libraries.add(match.group(1).split('.')[0])

# Walk through the project directory
for root, dirs, files in os.walk(project_dir):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Check for import and from statements
                    match = import_pattern.match(line) or from_pattern.match(line)
                    if match:
                        libraries.add(match.group(1).split('.')[0])
        elif file.endswith('.ipynb'):
            file_path = os.path.join(root, file)
            process_ipynb(file_path)

# Write libraries to requirements.txt
with open('requirements.txt', 'w') as req_file:
    for library in sorted(libraries):
        req_file.write(f"{library}\n")

print("Extracted libraries saved to requirements.txt")
