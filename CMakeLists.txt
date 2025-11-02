cmake_minimum_required(VERSION 3.12)
project(stock_market_simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

include_directories(cpp/include)

set(SOURCES
    cpp/src/mcmc_engine.cpp
    cpp/src/portfolio.cpp
    cpp/src/risk_analysis.cpp
    cpp/src/market_data.cpp
)

pybind11_add_module(stock_sim_core
    python/bindings/pybind_module.cpp
    ${SOURCES}
)

target_compile_definitions(stock_sim_core PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

if(MSVC)
    target_compile_options(stock_sim_core PRIVATE /O2)
else()
    target_compile_options(stock_sim_core PRIVATE -O3 -march=native)
endif()