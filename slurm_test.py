import subprocess
import re


def parse_gpu_info(node_info):
    """Extract total and allocated GPU counts from node info."""
    gres_match = re.search(r'Gres=.*gpu:(\d+)', node_info)
    alloc_match = re.search(r'AllocTRES=.*gres/gpu=(\d+)', node_info)

    total = int(gres_match.group(1)) if gres_match else 0
    allocated = int(alloc_match.group(1)) if alloc_match else 0
    free = total - allocated
    return total, allocated, free


def get_gpu_nodes():
    """Returns a list of HGX nodes with GPU info, excluding drained nodes."""
    result = subprocess.run(['scontrol', 'show', 'nodes'], capture_output=True, text=True)
    node_blocks = result.stdout.split('\n\n')  # each node block is separated by an empty line

    gpu_nodes = []

    for block in node_blocks:
        node_match = re.search(r'NodeName=(\S+)', block)
        state_match = re.search(r'State=(\S+)', block)

        if node_match:
            node_name = node_match.group(1)
            node_state = state_match.group(1) if state_match else ""

            # Skip if not HGX or if in DRAIN state
            if 'hgx' not in node_name.lower():
                continue
            if 'drain' in node_state:
                continue
            if 'invalid' in node_state:
                continue

            total, alloc, free = parse_gpu_info(block)
            if total > 0:
                gpu_nodes.append((node_name, total, alloc, free))

    return gpu_nodes


def find_least_occupied_gpu_node():
    gpu_nodes = get_gpu_nodes()
    if not gpu_nodes:
        print("No available HGX GPU nodes found.")
        return None

    # Sort by most free GPUs (descending)
    gpu_nodes.sort(key=lambda x: x[3], reverse=True)
    best_nodes = gpu_nodes[:20]
    return best_nodes


if __name__ == '__main__':
    node_info = find_least_occupied_gpu_node()
    if node_info:
        for node in node_info:
            node, total, alloc, free = node
            print(f"Node: {node} (Free: {free}, Allocated: {alloc}, Total: {total})")
