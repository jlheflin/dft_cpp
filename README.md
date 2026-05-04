# dft_cpp


## Dependencies

CMake should be able to find and install these suckers when needed:

- Eigen3
- Libint2
- Libxc
- fmt
- spdlog
- tbb
- OpenBLAS
  - Possibly OpenMP as well, if OpenBLAS is compiled with OpenMP support
- `sphere_lebedev_rule.cpp` and `sphere_lebedev_rule.hpp` from [FSU](https://people.sc.fsu.edu/~jburkardt/c_src/sphere_lebedev_rule/sphere_lebedev_rule.html)
  - Information on their page is distributed under the MIT License
  
## Build instructions

Clone:

```bash
  git clone https://github.com/jlheflin/dft_cpp.git
```

Configure:

```bash
  cd dft_cpp
  cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX=$PWD/install
```

Build:
```bash
  cmake --build build --parallel <desired_number_of_cores> --target install
```

## Running the code

Once the build is finished, you can run

```bash
 LD_PRELOAD=$PWD/install/lib ./build/dft_code <desired_basis_set_default_sto-3g>
```
