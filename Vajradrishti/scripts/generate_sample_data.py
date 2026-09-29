import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vajradrishti.synthetic import demo_replay
p=argparse.ArgumentParser(description="Generate synthetic event replay data")
p.add_argument("--output",default="data/synthetic/demo_event.json")
p.add_argument("--seed",type=int,default=42)
args=p.parse_args(); target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps({"data_mode":"SYNTHETIC / DEMO","events":demo_replay(seed=args.seed)},indent=2),encoding="utf-8")
print(f"Wrote {target} (synthetic demo data)")
