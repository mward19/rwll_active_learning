import yaml
import sys

base_config = sys.argv[1]
out_config = sys.argv[2]
nn_layer = int(sys.argv[3])
nn_update = int(sys.argv[4])

with open(base_config) as f:
    cfg = yaml.safe_load(f)

# Modify only what we care about
cfg["representation"]["nn_layer"] = nn_layer
cfg["representation"]["nn_update_interval"] = nn_update

with open(out_config, "w") as f:
    yaml.safe_dump(cfg, f)