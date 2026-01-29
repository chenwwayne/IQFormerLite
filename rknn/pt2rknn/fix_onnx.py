import onnx
import sys

def fix_transpose_perm(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    graph = model.graph
    
    fixed_count = 0
    for node in graph.node:
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
                    # perm is list of ints
                    ints = list(attr.ints)
                    rank = len(ints)
                    new_ints = []
                    changed = False
                    for i in ints:
                        if i < 0:
                            new_ints.append(i + rank)
                            changed = True
                        else:
                            new_ints.append(i)
                    
                    if changed:
                        print(f"Fixing Transpose node {node.name}: perm {ints} -> {new_ints}")
                        del attr.ints[:]
                        attr.ints.extend(new_ints)
                        fixed_count += 1
                        
    if fixed_count > 0:
        onnx.save(model, output_path)
        print(f"Saved fixed model to {output_path}")
    else:
        # Save anyway just to be sure we have a clean file or maybe just copy?
        # Better to save it if we want to ensure it's valid, but if no change, maybe just copy.
        # But let's overwrite if requested.
        print("No Transpose nodes with negative perm found. Saving copy anyway.")
        onnx.save(model, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_onnx.py input_model.onnx output_model.onnx")
        sys.exit(1)
    
    fix_transpose_perm(sys.argv[1], sys.argv[2])
