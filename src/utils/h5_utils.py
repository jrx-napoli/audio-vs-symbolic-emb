import h5py


def print_h5_structure(file_path):
    def recurse(name, obj, indent=""):
        if isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")
            for key, val in obj.attrs.items():
                print(f"{indent}  Attr: {key} = {val}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name} shape={obj.shape} dtype={obj.dtype}")
            for key, val in obj.attrs.items():
                print(f"{indent}  Attr: {key} = {val}")

    with h5py.File(file_path, "r") as f:
        f.visititems(lambda name, obj: recurse(name, obj))
