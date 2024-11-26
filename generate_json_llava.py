import json
import os

def find_image_path(base_folder, study_id):
    """
    Constructs the path to the image file based on the study_id.

    Args:
        base_folder (str): The root directory containing the study images.
        study_id (str): The study ID from the JSON value field.

    Returns:
        str: The full path to the image file, or an empty string if not found.
    """
    # Construct the study folder path
    study_folder = f"s{study_id}"
    study_path = os.path.join(base_folder, study_folder)

    # Look for the .jpg file in the study folder
    if os.path.exists(study_path):
        for file_name in os.listdir(study_path):
            if file_name.endswith(".jpg"):
                return os.path.join(study_path, file_name)
    return ""

def convert_to_llava_format_with_images(input_json_file, base_folder):
    """
    Converts the original JSON file into LLAVA-ready JSON format, including the image path,
    and filters only instances with study_id. Instances are renumbered sequentially.

    Args:
        input_json_file (str): Path to the original JSON file.
        base_folder (str): The root folder containing the study images.
    """
    # Generate the output file name dynamically
    output_json_file = os.path.splitext(input_json_file)[0] + "_processed.json"

    # Read the original JSON file
    with open(input_json_file, 'r') as f:
        original_data = json.load(f)
    
    llava_data = []
    instance_id = 1  # Sequential numbering starting from 1

    for entry in original_data:
        # Get study_id from the value field
        value_field = entry.get("value", {})
        study_id = str(value_field.get("study_id", ""))
        
        # Skip instances without a study_id
        if not study_id:
            continue
        
        # Find the corresponding image path
        image_path = find_image_path(base_folder, study_id)
        
        # Skip instances where the image path is not found
        if not image_path:
            continue

        # Process the answer field
        answer_value = entry.get("answer", [])
        if isinstance(answer_value, list):
            if len(answer_value) == 1 and answer_value[0] == 1:
                answer_value = "True"
            elif len(answer_value) == 1 and answer_value[0] == 0:
                answer_value = "False"
            elif not answer_value:  # Handle empty lists
                answer_value = " "
            else:
                answer_value = ", ".join(map(str, answer_value))  # Combine list into a string

        # Create LLAVA-ready entry
        llava_entry = {
            "id": instance_id,  # Sequential numbering
            "image": image_path,  # Include the extracted image path
            "conversations": [
                {
                    "from": "human",
                    "value": entry.get("template", "")  # Use "template" for human question
                },
                {
                    "from": "gpt",
                    "value": answer_value  # Use processed "answer" field for GPT response
                }
            ]
        }
        llava_data.append(llava_entry)
        instance_id += 1  # Increment the instance counter

    # Write the LLAVA-ready JSON to a new file
    with open(output_json_file, 'w') as f:
        json.dump(llava_data, f, indent=4)

    print(f"Converted JSON saved to {output_json_file}. {len(llava_data)} instances recorded.")

# Example usage
convert_to_llava_format_with_images(
    "/user_data/amulyam/Projects/EHRXQA/ehrxqa/dataset/mimic_iv_cxr/test.json", 
    "/user_data/amulyam/Projects/EHRXQA/ehrxqa/study_images"
)
