from os.path import join, exists, split, isdir
from os import listdir, mkdir, sep


def maybe_create_path(path):

    if path:
        counter = 0
        subfs = []
        bp = path

        while (not exists(bp)) and counter < 100:
            if bp.find(sep) >= 0:
                bp, subf = split(bp)
                subfs.append(subf)
            else:
                break

        if len(subfs) > 0:

            for subf in subfs[::-1]:
                mkdir(join(bp, subf))
                bp = join(bp, subf)
        else:
            if not exists(bp):
                mkdir(bp)


def my_listdir(path, return_pathes=False):
    content = listdir(path)
    content.sort()
    if return_pathes:
        return [join(path, cont) for cont in content]
    else:
        return content
