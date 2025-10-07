import os


def sort_ckp_paths(file_list, reverse=False):
    required_names = ["final", "best"]

    epoch2path = []
    name2path = {}
    for path in file_list:
        name = os.path.basename(path).split('.ckpt')[0].split('_')[-1]
        if name.isdigit():
            # epoch number checkpoint
            epoch = int(name)
            epoch2path.append((epoch, path))
        else:
            # final / best checkpoint
            name2path[name] = path

    sorted_paths = sorted(epoch2path, key=lambda x: x[0])
    sorted_paths = [path for _, path in sorted_paths]

    if reverse:
        sorted_paths = sorted_paths[::-1]

    for name in required_names:
        if name in name2path:
            sorted_paths.append(name2path[name])

    return sorted_paths

