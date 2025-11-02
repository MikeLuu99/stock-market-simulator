from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

ext_modules = [
    Pybind11Extension(
        "stock_sim_core",
        [
            "python/bindings/pybind_module.cpp",
            "cpp/src/mcmc_engine.cpp",
            "cpp/src/portfolio.cpp",
            "cpp/src/risk_analysis.cpp",
            "cpp/src/market_data.cpp",
        ],
        include_dirs=[
            "cpp/include",
            pybind11.get_include(),
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="stock-market-simulator",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)