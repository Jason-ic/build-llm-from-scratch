import json
import os

# data_path = os.path.join(os.path.dirname(__file__), "instruction-data.json")
# with open(data_path, "r") as f:
#     data = json.load(f)

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text

# train_portion = int(len(data) * 0.85)
# test_portion = int(len(data) * 0.1)
# val_portion = len(data) - train_portion - test_portion 

# train_data = data[:train_portion]
# test_data = data[train_portion:train_portion + test_portion]
# val_data = data[train_portion + test_portion:]
# print("Training set length:", len(train_data))
# print("Validation set length:", len(val_data))
# print("Test set length:", len(test_data))