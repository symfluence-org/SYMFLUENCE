#!/usr/bin/env python3
"""Generate FUSE and GR4J configs for all 59 LamaH-Ice catchments.

Usage:
    python generate_configs.py --template _template_fuse.yaml --output ./generated/fuse/
    python generate_configs.py --template _template_gr4j.yaml --output ./generated/gr4j/

The DOMAIN_ID placeholder in the template is replaced with each catchment ID.
Bounding box coordinates are read from the LamaH-Ice catchment attributes.
"""
import argparse
import os

# LamaH-Ice catchment IDs used in the paper (59 catchments)
CATCHMENT_IDS = [
    "102", "104", "108", "110", "112", "114", "119", "120",
    "126", "128", "132", "135", "138", "139", "146", "148",
    "150", "152", "155", "160", "162", "165", "168", "170",
    "172", "176", "180", "182", "186", "188", "190", "192",
    "196", "198", "200", "202", "204", "208", "210", "212",
    "216", "218", "220", "224", "226", "228", "230", "232",
    "1001", "1002", "1005", "1006", "1007", "1008", "1009",
    "1010", "1012", "1015", "1016"
]

def generate_configs(template_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(template_path) as f:
        template = f.read()

    for cid in CATCHMENT_IDS:
        config = template.replace("DOMAIN_ID", cid)
        model = "FUSE" if "fuse" in template_path.lower() else "GR4J"
        fname = f"config_lamahice_{cid}_{model}.yaml"
        with open(os.path.join(output_dir, fname), 'w') as f:
            f.write(config)
    print(f"Generated {len(CATCHMENT_IDS)} configs in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_configs(args.template, args.output)
