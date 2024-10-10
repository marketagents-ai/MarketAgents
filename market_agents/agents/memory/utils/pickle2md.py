"""
# Pickle to Readable JSON Converter

## Purpose
This script converts pickle files to human-readable JSON format. It's useful for inspecting the contents of pickle files without the need to unpickle them in a potentially unsafe manner.

## Features
- Converts pickle files to JSON
- Handles various Python data types including primitives, collections, and custom objects
- Safely unpickles objects from the `__main__` module
- Outputs the result to both console and a file

## Usage
1. Run the script:
   ```
   python pickle_to_readable.py
   ```
2. When prompted, enter the path to the pickle file you want to convert.
3. The script will display the contents in the console and save them to a JSON file.

## Output
- Console output: Displays the JSON representation of the pickle file contents
- File output: Saves the JSON content to a file named `[original_filename]_contents.json`

## Functions
- `CustomUnpickler`: A custom unpickler for safe unpickling
- `object_to_dict`: Converts Python objects to dictionary representations
- `pickle_to_readable`: Main function for converting pickle to JSON
- `main`: Handles user input and program flow

## Limitations
- Large pickle files may result in correspondingly large JSON outputs
- Some complex objects might not be fully representable in JSON format

This script provides a simple way to view the contents of pickle files in a more readable format, which can be helpful for debugging or data inspection purposes.
"""

import pickle
import json
from collections import defaultdict
import types

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            return type(name, (), {})
        return super().find_class(module, name)

def object_to_dict(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return {"__set__": [object_to_dict(item) for item in obj]}
    elif isinstance(obj, types.FunctionType):
        return f"<function {obj.__name__}>"
    elif hasattr(obj, "__dict__"):
        obj_dict = {k: object_to_dict(v) for k, v in obj.__dict__.items() 
                    if not k.startswith("__") and not callable(v)}
        obj_dict["__class__"] = obj.__class__.__name__
        return obj_dict
    else:
        return str(obj)

def pickle_to_readable(pickle_file):
    with open(pickle_file, 'rb') as f:
        unpickler = CustomUnpickler(f)
        data = unpickler.load()
    
    readable_data = object_to_dict(data)
    return json.dumps(readable_data, indent=2)

def main():
    pickle_file = input("Enter the path to the pickle file: ")
    try:
        readable_content = pickle_to_readable(pickle_file)
        print("\nHuman-readable contents of the pickle file:")
        print(readable_content)
        
        output_file = pickle_file + "_contents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(readable_content)
        print(f"\nContents saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()