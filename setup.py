from distutils.core import setup
from setuptools import Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "sorted_nearest.src.sorted_nearest",
        ["sorted_nearest/src/sorted_nearest.pyx"],
    ),
    Extension(
        "sorted_nearest.src.max_disjoint_intervals",
        ["sorted_nearest/src/max_disjoint_intervals.pyx"],
    ),
    Extension(
        "sorted_nearest.src.k_nearest",
        ["sorted_nearest/src/k_nearest.pyx"],
    ),
    Extension(
        "sorted_nearest.src.k_nearest_ties",
        ["sorted_nearest/src/k_nearest_ties.pyx"],
    ),
    Extension(
        "sorted_nearest.src.clusters",
        ["sorted_nearest/src/clusters.pyx"],
    ),
    Extension(
        "sorted_nearest.src.annotate_clusters",
        ["sorted_nearest/src/annotate_clusters.pyx"],
    ),
    Extension(
        "sorted_nearest.src.cluster_by",
        ["sorted_nearest/src/cluster_by.pyx"],
    ),
    Extension(
        "sorted_nearest.src.merge_by",
        ["sorted_nearest/src/merge_by.pyx"],
    ),
    Extension(
        "sorted_nearest.src.introns",
        ["sorted_nearest/src/introns.pyx"],
    ),
    Extension(
        "sorted_nearest.src.windows",
        ["sorted_nearest/src/windows.pyx"],
    ),
    Extension(
        "sorted_nearest.src.tiles",
        ["sorted_nearest/src/tiles.pyx"],
    ),
]

setup(ext_modules=cythonize(extensions, language_level=3))
