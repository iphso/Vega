import json

from datasets import load_dataset

ds = load_dataset("proxima-fusion/constellaration", split="train", streaming=True)
row = next(iter(ds))


def describe(obj, prefix="", depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                describe(v, path, depth + 1, max_depth)
            elif isinstance(v, list):
                shape = []
                cur = v
                while isinstance(cur, list):
                    shape.append(len(cur))
                    cur = cur[0] if cur else None
                print(f"{path}: list shape={shape} inner_type={type(cur).__name__}")
            else:
                print(f"{path}: {type(v).__name__} = {v!r:.80}")
    else:
        print(f"{prefix}: {type(obj).__name__} = {obj!r:.80}")


print("=== TOP LEVEL KEYS ===")
print(list(row.keys()))
print()
print("=== FULL STRUCTURE (depth-limited) ===")
describe(row)
