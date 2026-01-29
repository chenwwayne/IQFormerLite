import onnx
import sys
import os

def fix_transpose_perm(model_path, output_path=None):
    print(f"Checking model for Transpose nodes with negative perm: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph
    
    fixed_count = 0
    for node in graph.node:
        if node.op_type == "Transpose":
            for attr in node.attribute:
                if attr.name == "perm":
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
                        
    if output_path is None:
        base, ext = os.path.splitext(model_path)
        output_path = f"{base}_fixed{ext}"

    if fixed_count > 0:
        onnx.save(model, output_path)
        print(f"Saved fixed model to {output_path}")
        return output_path
    else:
        print("No fixes needed. Saving copy anyway to ensure workflow consistency.")
        onnx.save(model, output_path)
        return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_onnx.py input_model.onnx [output_model.onnx]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_transpose_perm(input_path, output_path)
