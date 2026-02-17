import os
from typing import List


def _get_all_from_init(init_path: os.PathLike) -> List[str]:
    """Extract __all__ list from a __init__.py file"""
    with open(init_path, "r") as f:
        content = f.read()

    # Check if __all__ is defined
    if "__all__" not in content:
        return []

    # Remove comments, whitespace, and newlines
    content = "\n".join(
        [line for line in content.splitlines() if not line.lstrip().startswith("#")]
    )
    content = content.replace(" ", "").replace("\n", "")

    # Extract __all__ list
    all_list = content.split("__all__=[")[1].split("]")[0].split(",")
    all_list = [item.replace("'", "").replace('"', "") for item in all_list if item]

    return all_list


def _build_structure(
    root_dir: os.PathLike, current_dir: os.PathLike, structure_dict: dict
) -> None:
    """Recursively build the structure dictionary"""
    try:
        items = os.listdir(current_dir)
    except:
        return

    # Check for __init__.py
    if "__init__.py" in items:
        all_list = _get_all_from_init(os.path.join(current_dir, "__init__.py"))
        if all_list:
            rel_path = os.path.relpath(current_dir, root_dir)
            if rel_path == ".":
                key = "wah"
            else:
                key = rel_path
            structure_dict[key] = all_list

    # Recursively process subdirectories
    for item in sorted(items):
        item_path = os.path.join(current_dir, item)
        if (
            os.path.isdir(item_path)
            and not item.startswith(".")
            and not item.startswith("__")
        ):
            _build_structure(root_dir, item_path, structure_dict)


def build_nested_structure(structure_dict, keys, current_path=""):
    result = []
    for item in structure_dict.get(current_path if current_path else "wah", []):
        next_path = f"{current_path}/{item}" if current_path else item
        if next_path in keys:
            result.append(
                {item: build_nested_structure(structure_dict, keys, next_path)}
            )
        else:
            result.append(item)
    return result


# Convert nested_dict to markdown
def dict_to_md(d, level=0):
    lines = []
    for key, value in d.items():
        # Add appropriate number of # based on level
        prefix = "\t" * level
        lines.append(f"{prefix}- `{key}`")

        if isinstance(value, list):
            # For list items, use bullet points
            for item in value:
                if isinstance(item, dict):
                    # Handle nested dict within list
                    lines.extend(dict_to_md(item, level + 1))
                else:
                    lines.append(f"{prefix}\t- `{item}`")
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries with increased indentation
            nested_lines = dict_to_md(value, level + 1)
            lines.extend(nested_lines)
    return lines


def create_structure_md(root_dir: os.PathLike = "src/wah") -> str:
    # Build structure dictionary
    structure_dict = {}
    _build_structure(root_dir, root_dir, structure_dict)
    keys = list(structure_dict.keys())
    nested_dict = {}
    nested_dict["wah"] = build_nested_structure(structure_dict, keys)

    # Generate markdown content
    md_lines = dict_to_md(nested_dict)[1:]
    md_lines = [line[1:] for line in md_lines]
    structure_md = "### `wah`\n" + "\n".join(md_lines)

    return structure_md


if __name__ == "__main__":
    # Generate the structure and save to file
    structure_md = create_structure_md()

    with open("structure.md", "w") as f:
        f.write(structure_md)
