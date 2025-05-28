#!/usr/bin/env python3
# Copyright 2025 FBK
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import json
import pandas as pd


tsv_path = ""  # Path to the tsv file containing data information
output_file = ""  # Output json file
annotation = ""  # "words" to merge orthographic annotation, "phones" for phonetic annotation

data = pd.read_csv(tsv_path, sep='\t')
all_files = data[annotation]

linguistic_data = {}
for i, file_path in enumerate(all_files):
    with open(file_path, "r") as f:
        annotations = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Ensure valid line format
                start, end, linguistic_unit = int(parts[0]), int(parts[1]), parts[2]
                annotations.append((start, end, linguistic_unit))
        linguistic_data[i] = annotations  # Use the filename without the extension as the key

with open(output_file, "w") as f:
    json.dump(linguistic_data, f, indent=4)

print(f"Data saved to {output_file}")
