

import os
import numpy as np

def should_skip_line(lines):
    """
    Check if the line should be skipped based on the presence of 'delete' in any file.
    """
    return any('delete' in line for line in lines)

def process_files(folder_path):
    """
    Process all txt files in the given folder.
    """
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    all_lines = [[] for _ in range(2767)]  # Assuming each file has 2767 lines

    # Read and store all lines from all files
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file):
                all_lines[line_num].append(line.strip())

    # Filter out lines with 'delete'
    valid_lines = [lines for lines in all_lines if not should_skip_line(lines)]

    # Process valid lines for labels
    labels_1, labels_2 = [], []
    for lines in valid_lines:
        line_labels_1, line_labels_2 = [], []
        for line in lines:
            _, labels_str = line.split(':')
            label1, label2 = map(int, labels_str.split(','))
            line_labels_1.append(label1)
            line_labels_2.append(label2)
        # Calculate mean and round for valid lines
        labels_1.append(round(np.mean(line_labels_1)))
        labels_2.append(round(np.mean(line_labels_2)))

    # Write results to new files
    with open('gt_1.txt', 'w') as gt1, open('gt_2.txt', 'w') as gt2:
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            if i < len(labels_1):  # Check to avoid index error
                gt1.write(f'{file_name}:{labels_1[i]}\n')
                gt2.write(f'{file_name}:{labels_2[i]}\n')
            
def validate_files(folder_path, expected_lines=2767):
    """
    Validate that each file in the folder has the expected number of lines
    and check if the lines across files are consistent in terms of file names.
    """
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    line_contents = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) != expected_lines:
                raise ValueError(f"File {os.path.basename(file_path)} does not have {expected_lines} lines.")
            
            for line_num, line in enumerate(lines):
                file_name, _ = line.strip().split(':', 1)
                if line_num not in line_contents:
                    line_contents[line_num] = file_name
                elif line_contents[line_num] != file_name:
                    raise ValueError(f"File name mismatch in line {line_num+1} between files: {file_path} and previous files. Expected '{line_contents[line_num]}', found '{file_name}'.")
def main():
    folder_path = 'raw_label'  # Adjust this to your folder path
    validate_files(folder_path)
    process_files(folder_path)

if __name__ == '__main__':
    main()

