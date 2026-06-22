// Instead of using the internal implementation, we use external BLAS libraries
#include <deque>
#define EIGEN_USE_BLAS
// The "NumPy" of C++
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
// I watched a user named "magicalbat" on YouTube use cstdint to define
// different size integers and unsigned integers, I liked the idea so
// I copied it.
#include <cstdint>
// spdlog with libfmt is so much better than using std::cout
#include <spdlog/spdlog.h>
// std::functional is used to definte an arbitrary function and its return
// types, it was like something I did in Python for creating a dict for
// calling different functions
#include <functional>
// Purely an AI suggestion, enables multithreaded execution for the Becke
// partition evaluation so that it doesn't take forever
#include <execution>
// Provides the one- and two-electron integrals needed to actually run DFT
// (S, T, V_ne, V_ee)
#include <libint2.hpp>
// Provides a C interface to the different DFT functionals. From what I have
// seen so far, C and Fortran based codes have an "allocate memory then fill"
// paradigm, which I mimic with some functions
#include <xc.h>
// Provides the Lebedev angular grids!
#include "sphere_lebedev_rule.hpp"
// A timing utility, used to providing timing data between steps
#include <chrono>


using clk = std::chrono::high_resolution_clock;
auto t0 = clk::now();
auto elapsed = [](auto start) {
  return std::chrono::duration<double>(clk::now() - start).count();
};

// Apparently this is the same naming scheme as Rust, essentially this is
// just renaming the types to be more usable.
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

// Same thing, assigning a "type" to a name for easy use later
using LebedevFunc = std::function<void(double*, double*, double*, double*)>;

// Essentially a Python dictionary, where I put in a specific integer, and get the
// related function out.
std::unordered_map<int, LebedevFunc> lebedev_map = {
    {6,   ld0006},
    {14,  ld0014},
    {26,  ld0026},
    {38,  ld0038},
    {50,  ld0050},
    {74,  ld0074},
    {86,  ld0086},
    {110, ld0110},
    {146, ld0146},
    {170, ld0170},
    {194, ld0194},
    {302, ld0302},
    {590, ld0590},
    {974, ld0974},
    {1454, ld1454},
};

// Remember how I said that I experimented with the same C/Fortran interface rules?
// This is one of those functions, the same can be said for two_e_engine_run as well.
// Just calls libint and fills the Eigen matrix
void one_e_engine_run(std::string type, libint2::BasisSet obs, Eigen::MatrixXd& matrix, std::vector<libint2::Atom> atoms) {
  libint2::initialize();
  libint2::Operator op;
  if (type == "kinetic") {
    op = libint2::Operator::kinetic;
  } else if (type == "overlap") {
    op = libint2::Operator::overlap;
  } else {
    op = libint2::Operator::nuclear;
  }

  libint2::Engine engine(op, obs.max_nprim(), obs.max_l());

  if (type == "nuclear") {
    engine.set_params(libint2::make_point_charges(atoms));
  }

  const auto& shell_sets = obs.shell2bf();

  for (uint s1 = 0; s1 < obs.size(); s1++) {
    for (uint s2 = 0; s2 < obs.size(); s2++) {
      engine.compute(obs[s1], obs[s2]);
      const auto* buf = engine.results()[0];
      if (buf == nullptr) {
        continue;
      }

      uint bf1 = shell_sets[s1];
      uint bf2 = shell_sets[s2];
      uint n1  = obs[s1].size();
      uint n2  = obs[s2].size();

      for (uint f1 = 0; f1 < n1; f1++) {
        for (uint f2 = 0; f2 < n2; f2++) {
          matrix(bf1 + f1, bf2 + f2) = buf[f1 * n2 + f2];
        }
      }
    }
  }
  libint2::finalize();
}

// So this is a little lost on me, but to save on memory I (with the help of Claude)
// exploited the fact that the electron repulsion integrals (ERIs) have 8-fold
// symmetry, so I only need to store 1/8 of the values. This function basically
// makes it easier to get the index of the associated symmetric value within the
// ERI tensor
int ten_idx(int i, int j, int k, int l) {
  if (i < j) {
    std::swap(i, j);
  }
  if (k < l) {
    std::swap(k, l);
  }

  int ij = i * (i + 1) / 2 + j;
  int kl = k * (k + 1) / 2 + l;

  if (ij < kl) {
    std::swap(ij, kl);
  }

  return ij * (ij + 1) / 2 + kl;
}

void two_e_engine_run(libint2::BasisSet obs, Eigen::VectorXd& vector) {
  libint2::initialize();
  libint2::Engine engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());
  const auto& shell_sets = obs.shell2bf();
  int N = obs.nbf();

  for (uint s1 = 0; s1 < obs.size(); s1++) {
    for (uint s2 = 0; s2 <= s1; s2++) {
      uint ss12 = s1 * (s1 + 1) / 2 + s2;
      for (uint s3 = 0; s3 < obs.size(); s3++) {
        for (uint s4 = 0; s4 <= s3; s4++) {
          uint ss34 = s3 * (s3 + 1) / 2 + s4;
          if (ss34 > ss12) {
            continue;
          }

          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
          const auto* buf = engine.results()[0];
          if (buf == nullptr) {
            continue;
          }

          int bf1 = shell_sets[s1], n1 = obs[s1].size();
          int bf2 = shell_sets[s2], n2 = obs[s2].size();
          int bf3 = shell_sets[s3], n3 = obs[s3].size();
          int bf4 = shell_sets[s4], n4 = obs[s4].size();

          for (uint f1 = 0; f1 < n1; f1++) {
            for (uint f2 = 0; f2 < n2; f2++) {
              for (uint f3 = 0; f3 < n3; f3++) {
                for (uint f4 = 0; f4 < n4; f4++) {
                  f64 val = buf[f1*n2*n3*n4 + f2*n3*n4 + f3*n4 + f4];
                  int i = bf1 + f1, j = bf2 + f2, k = bf3 + f3, l = bf4 + f4;
                  vector(ten_idx(i,j,k,l)) = val;
                }
              }
            }
          }
        }
      }
    }
  }
  libint2::finalize();
}

// This is where the fun begins!
// argc is the integer count of the number of arguments provided to readline,
// which I believe is the program that literally "reads" the command prompt
// when you press enter. I also believe this is always greater than or equal
// to one, so if you run "./dft_code", the argument count is 1. The argv[]
// object is the array storage of what readline "reads", so if I submit
// something like "./dft_code test", then argc would be 2, and argv[0] would
// be "./dft_code" and argv[1] would be "test"
int main(int argc, char* argv[]) {
  auto t_main = clk::now();

  // This sets what gets "printed" to stdout later on in the code.
  // If I only want stuff that is labeled "info" I would set this to
  // spdlog::level::info
  spdlog::set_level(spdlog::level::trace);
  spdlog::info("Intitializing variables...");

  // Gotta tell C++ that you will eventually have data to store, specifically the
  // basis set you want to use in this case, which I have assigned as std::string.
  // If I don't provide a basis in the command line, I just go ahead and set it to
  // "sto-3g"
  // I actually was experimenting with a "allocate all memory at the beginning"
  // scheme, which is similar to how embedded systems have their code set up
  // (think car compuers and such). This doesn't really work out since some
  // variables are only needed one to build something, like the "final_weights"
  // object, which just wastes memory
  std::string basis;
  if (argc < 2) {
    basis = "sto-3g";
  } else {
    basis = argv[1];
  }
  spdlog::trace("Intitializing libint2...");

  // Boilerplate code, SOP for libint
  // This is the file I plan on reading in using libint. I am too lazy to change the
  // filename each time and recompile, so I just changed the contents of the file.
  std::string filename = "./structure.xyz";
  std::ifstream input_file(filename);
  std::vector<libint2::Atom> atoms;
  if (input_file.is_open()) {
    std::string xyz((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
    spdlog::info("Using structure located in {}", filename);
    spdlog::trace("XYZ:\n{}", xyz);

    std::istringstream iss(xyz);
    atoms = libint2::read_dotxyz(iss);

  } else {
    spdlog::info("Using predefined H2");
    spdlog::trace("XYZ:\n2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 2.0");

    libint2::Atom H1{1, 0.0, 0.0, 0.0};
    libint2::Atom H2{1, 0.0, 0.0, 2.0};
    atoms.push_back(H1);
    atoms.push_back(H2);
  }

  libint2::BasisSet obs(basis, atoms);
  // Boilerplate code, SOP for libint
  libint2::Engine S_engine(libint2::Operator::overlap, obs.max_nprim(), obs.max_l());
  libint2::Engine T_engine(libint2::Operator::kinetic, obs.max_nprim(), obs.max_l());
  libint2::Engine V_ne_engine(libint2::Operator::nuclear, obs.max_nprim(), obs.max_l());
  V_ne_engine.set_params(libint2::make_point_charges(atoms));
  libint2::Engine ERI_engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l());

  // This is where we start to see some similar parameters to that of codes
  // like NWChem.
  u64 radial_points    = 400;
  // This a and m are associated with the Mura-Knowles radial quadrature scheme.
  // The a in the original paper was 5.0, but I saw that in the PySCF code it was
  // set to 5.2 so I switched to that. Not sure why!
  f64 a                = 5.2;
  u8 m                = 3;

  // This k factor is associated with the Becke partitioning scheme, and is
  // what determines how many "fuzzy" passes we make on the partitions
  u8 k                = 3;

  // Number of angular points to use
  u64 n_ang            = 1454;
  u64 n_atoms         = atoms.size();
  u64 n_basis         = obs.nbf();

  // This is just summing up the number of electrons. Way too convoluted
  u64 n_elec          = std::accumulate(atoms.begin(), atoms.end(), 0,
                                        [](int sum, const libint2::Atom& atom)
                                      {return sum + atom.atomic_number;});

  // No SCF like Restricted SCF
  u64 n_occ = n_elec / 2;
  u64 points_per_atom = radial_points * n_ang;
  u64 total_points    = n_atoms * points_per_atom;

  // Now we get into similar values used in QC codes, like convergence
  // parameters and max iterations
  u64 max_iter        = 100;
  f64 conv_tol        = 1e-6;
  f64 E_old           = 0.0;
  f64 E_new           = 0.0;
  f64 E_nuc           = 0.0;
  f64 damping         = 0.3;
  u64 diis_size       = 6;
  bool converged      = false;

  // More allocation of objects, although all the "Xd" Eigen types
  // are dynamically allocated, so I'm not necessarily allocating the
  // memory (at least I don't think), so really this is just helpful
  // for being at the top
  Eigen::VectorXd i_vec(radial_points), x(radial_points);
  Eigen::VectorXd r(radial_points), drdx(radial_points);
  Eigen::VectorXd w(radial_points), W(radial_points);
  Eigen::MatrixXd l_points(n_ang, 3);
  Eigen::VectorXd l_weights(n_ang);
  Eigen::MatrixXd atom_pos(n_atoms, 3);
  Eigen::MatrixXd all_coords(total_points, 3);
  Eigen::VectorXd all_weights(total_points);

  Eigen::MatrixXd P_becke(total_points, n_atoms);
  Eigen::VectorXd P_sum(total_points);
  Eigen::MatrixXd bw(total_points, n_atoms);
  Eigen::VectorXd final_weights(total_points);

  Eigen::MatrixXd chi(total_points, n_basis);

  Eigen::MatrixXd P_new(n_basis, n_basis);
  Eigen::MatrixXd P_old(n_basis, n_basis);
  Eigen::VectorXd rho(total_points), exc(total_points), vxc(total_points);
  Eigen::MatrixXd V_xc(n_basis, n_basis);

  Eigen::MatrixXd S(n_basis, n_basis), T(n_basis, n_basis), V_ne(n_basis, n_basis);
  Eigen::MatrixXd J(n_basis, n_basis), K(n_basis, n_basis), F(n_basis, n_basis);
  Eigen::MatrixXd H(n_basis, n_basis);
  Eigen::MatrixXd C(n_basis, n_basis);
  u64 n_unique = n_basis * (n_basis + 1) / 2;
  u64 eri_size = n_unique * (n_unique + 1) / 2;
  Eigen::VectorXd ERI(eri_size);

  std::deque<Eigen::MatrixXd> fock_history;
  std::deque<Eigen::MatrixXd> error_history;

  spdlog::info("Creating radial points...");
  // Build radial points, this uses the Mura-Knowles scheme.
  i_vec = Eigen::VectorXd::LinSpaced(radial_points, 1, radial_points);
  x = i_vec / (radial_points + 1);
  r = -a * (1 - x.array().pow(m)).log();
  drdx = a * m * x.array().pow(m-1) / (1 - x.array().pow(m));
  w = drdx / (radial_points + 1);
  W = w.array() * r.array().pow(2);

  spdlog::info("Creating Lebedev points and weights...");
  // Build Lebedev quadrature points and weights
  lebedev_map[n_ang](l_points.col(0).data(), l_points.col(1).data(), l_points.col(2).data(), l_weights.data());
  l_weights *= 4.0 * M_PI;

  for (int i = 0; i < atoms.size(); i++) {
    atom_pos(i, 0) = atoms[i].x;
    atom_pos(i, 1) = atoms[i].y;
    atom_pos(i, 2) = atoms[i].z;
  }

  int idx = 0;
  for (int pos = 0; pos < atom_pos.rows(); pos++) {
    for (int i = 0; i < r.size(); i++) {
      for (int j = 0; j < l_weights.rows(); j++) {
        all_coords.row(idx) = r(i) * l_points.row(j) + atom_pos.row(pos);
        all_weights(idx) = W(i) * l_weights(j);
        idx++;
      }
    }
  }

  int n_points = all_coords.rows();

  spdlog::info("Applying Becke partitioning scheme...");
  // So the Becke partitioning scheme divies up points in the grid points to each
  // nuclei, if this isn't used the points are clearly "owned" by their own respective
  // nuclei that generated them, as opposed to maybe being more influenced by another nuclei.
  // This allows for greater flexiblity for the functional to work in.
  // rows = grid points, cols = atoms
  spdlog::info("Creating r_all...");
  Eigen::MatrixXd r_all(total_points, n_atoms);
  for (int atom_k = 0; atom_k < n_atoms; atom_k++) {
      r_all.col(atom_k) = (all_coords.rowwise() - atom_pos.row(atom_k)).rowwise().norm();
  }
  P_becke = Eigen::MatrixXd::Ones(total_points, n_atoms);

  std::vector<int> atom_indices(n_atoms);
  std::iota(atom_indices.begin(), atom_indices.end(), 0);
  spdlog::trace("Becke parallel execution starting...");
  std::for_each(std::execution::par_unseq, atom_indices.begin(), atom_indices.end(), [&](int i) {
    for (int j = 0; j < n_atoms; j++) {
      if (i == j) {
        continue;
      }

      auto R_ij = (atom_pos.row(i) - atom_pos.row(j)).norm();
      Eigen::ArrayXd mu = (r_all.col(i) - r_all.col(j)) / R_ij;
      for (int round = 0; round < k; round++) {
        mu = 1.5 * mu - 0.5 * mu * mu * mu;
      }
      P_becke.col(i).array() *= 0.5 * (1.0 - mu);
    }
  });

  spdlog::info("Computing P_sum...");
  P_sum = P_becke.rowwise().sum();
  spdlog::info("Computing bw...");
  bw = P_becke.array().colwise() / P_sum.array();

  spdlog::info("Creating final weights...");
  final_weights = Eigen::VectorXd::Zero(n_points);
  for (int atom_idx = 0; atom_idx < n_atoms; atom_idx++) {
    int start = atom_idx * points_per_atom;
    final_weights.segment(start, points_per_atom) =
      all_weights.segment(start, points_per_atom).array() *
      bw.block(start, atom_idx, points_per_atom, 1).array();
  }

  // Chi is the evaluation of the basis functions on the grid. We're basically plugging in
  // our r value for the function and getting out the result. This was actually one of the
  // hardest things to solve, there's a specific ordering to the functions in regards to the
  // angular portions and if you don't get it right, then everything is messed up.
  // The coefficients themself that we get back are normalized "to unity", which in
  // my understanding is "we have done everything we can with just the basis set info,
  // you must provide the coordinates now"
  auto t_chi_start = clk::now();
  spdlog::info("Creating chi...");
  u64 skipped = 0;
  chi = Eigen::MatrixXd::Zero(all_coords.rows(), n_basis);
  for (int s = 0; s < obs.size(); s++) {
    const auto& shell = obs[s];
    f64 alpha_min = *std::min_element(shell.alpha.begin(), shell.alpha.end());
    int bf = obs.shell2bf()[s];
    Eigen::Vector3d shell_origin(shell.O[0], shell.O[1], shell.O[2]);

    for (int row = 0; row < all_coords.rows(); row++) {
      auto diff = all_coords.row(row) - shell_origin.transpose();
      f64 r2 = diff.squaredNorm();
      if (exp(-alpha_min * r2) < 1e-12) {
        skipped += 1;
        continue;
      }
      auto x = diff(0);
      auto y = diff(1);
      auto z = diff(2);
      auto x2 = x * x;
      auto y2 = y * y;
      auto z2 = z * z;
      auto x3 = x2 * x;
      auto y3 = y2 * y;
      auto z3 = z2 * z;

      for (int p = 0; p < shell.nprim(); p++) {
        f64 prefactor = exp(-shell.alpha[p] * r2) * shell.contr[0].coeff[p];
        
        if (shell.contr[0].l == 0) {
          chi(row, bf) += prefactor;
        } else if (shell.contr[0].l == 1) {
          chi(row, bf + 0) += prefactor * x;
          chi(row, bf + 1) += prefactor * y;
          chi(row, bf + 2) += prefactor * z;
        } else if (shell.contr[0].l == 2) {
          if (shell.contr[0].pure) {
            chi(row, bf + 0) += prefactor * sqrt(3.0) * x * y;  // m=-2
            chi(row, bf + 1) += prefactor * sqrt(3.0) * y * z;  // m=-1
            chi(row, bf + 2) += prefactor * (z2 - (x2 + y2) / 2);  // m= 0
            chi(row, bf + 3) += prefactor * sqrt(3.0) * x * z;  // m=+1
            chi(row, bf + 4) += prefactor * sqrt(3.0) * (x2 - y2) / 2;  // m=+2
          } else {
            chi(row, bf + 0) += prefactor * x2;
            chi(row, bf + 1) += prefactor * x * y;
            chi(row, bf + 2) += prefactor * x * z;
            chi(row, bf + 3) += prefactor * y2;
            chi(row, bf + 4) += prefactor * y * z;
            chi(row, bf + 5) += prefactor * z2;
          }
        } else if (shell.contr[0].l == 3) {
          if (shell.contr[0].pure) {
            chi(row, bf + 0) += prefactor * sqrt(10.0) * ((3*x2*y - y3) / 4.0);
            chi(row, bf + 1) += prefactor * sqrt(15.0) * x * y * z;
            chi(row, bf + 2) += prefactor * sqrt(6.0) * (y*z2 - ((x2*y + y3) / 4.0));
            chi(row, bf + 3) += prefactor * (z3 - 3.0*(x2*z +y2*z) / 2);
            chi(row, bf + 4) += prefactor * sqrt(6.0) * (x*z2 - (x3 + x*y2) / 4.0);
            chi(row, bf + 5) += prefactor * sqrt(15.0) * (x2*z - y2*z) / 2;
            chi(row, bf + 6) += prefactor * sqrt(10.0) * (x3 - 3*x*y2) / 4;
          } else {
            chi(row, bf + 0) += prefactor * x3;  // m=-3
            chi(row, bf + 1) += prefactor * x2 * y;  // m=-2
            chi(row, bf + 2) += prefactor * x2 * z;  // m=-1
            chi(row, bf + 3) += prefactor * x * y2; // m= 0
            chi(row, bf + 4) += prefactor * x * y * z;  // m=+1
            chi(row, bf + 5) += prefactor * x * z2;  // m=+2
            chi(row, bf + 6) += prefactor * y3;  // m=+3
            chi(row, bf + 7) += prefactor * y2 * z;  // m=+3
            chi(row, bf + 8) += prefactor * y * z2;  // m=+3
            chi(row, bf + 9) += prefactor * z3;  // m=+3
          }
        }
      }
    }
  }
  spdlog::trace("Number of shells skipped: {}", skipped);
  spdlog::info("Chi build took: {:.3f}s", elapsed(t_chi_start));
  f64 threshold = 1e-12;
  u64 n_zero = (chi.array().abs() < threshold).count();
  u64 n_total = chi.size();
  f64 sparsity = 100.0 * n_zero / n_total;
  spdlog::info("chi sparsity: {}/{} ({:.2f}%) below {:.0e}", n_zero, n_total, sparsity, threshold);
  

  spdlog::info("Initializing Libxc functional...");
  // Libxc boilerplate code, the "1" is similar to the mapping as above
  // with the Lebedev angular points, this 1 is the id for the slater/lda functional.
  // You can see the func_id on libxc's site
  xc_func_type func;
  if (xc_func_init(&func, 1, XC_UNPOLARIZED) != 0) {
    throw std::runtime_error("Failed to initialize libxc functional");
  }


  spdlog::info("Populating S, T, V_ne, ERI from Libint2...");
  one_e_engine_run("overlap", obs, S, atoms);
  one_e_engine_run("kinetic", obs, T, atoms);
  one_e_engine_run("nuclear", obs, V_ne, atoms);
  auto t_eri_start = clk::now();
  two_e_engine_run(obs, ERI);
  spdlog::info("ERI build took: {:.3f}s", elapsed(t_eri_start));
  spdlog::trace("Finalizing libint2...");

  spdlog::info("Creating H_core...");
  H = T + V_ne;
  // spdlog::info("Creating GWH guess...");
  // constexpr f64 K_gwh = 1.75;
  // Eigen::MatrixXd F_guess(n_basis, n_basis);
  // for (int u = 0; u < n_basis; u++) {
  //   F_guess(u, u) = H(u, u);
  //   for (int v = 0; v < u; v++) {
  //     f64 val = 0.5 * K_gwh * S(u, v) * (H(u, u) + H(v, v));
  //     F_guess(u, v) = val;
  //     F_guess(v, u) = val;
  //   }
  // }
  

  // This piece of code was a great diagnostic to see if my chi was being built
  // correctly. I need to better understand the math of how it works, but if
  // S_grid != S, then there was a problem.
  // Eigen::MatrixXd S_grid = chi.transpose() * final_weights.asDiagonal() * chi;
  // Eigen::MatrixXd S_diff = (S_grid.array().abs() - S.array().abs());
  // S_diff = (S_diff.array() < 1e-10).select(0.0, S_diff);
  // std::cout << "S_diff:\n" << S_diff << std::endl;

  spdlog::info("Solving guess Hamiltonian...");
  // This gets us our eigenvalues and eigenvectors, but we just use the eigenvectors
  // for now
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> guess(H, S);
  spdlog::info("Creating C and P_new...");
  C = guess.eigenvectors();
  P_new = 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();

  spdlog::info("Starting DFT SCF loop...");
  while (not converged && max_iter-- > 0) {
    auto t_j = clk::now();
    spdlog::trace("Building J (Iter {})...", (100 - max_iter));
    J = Eigen::MatrixXd::Zero(n_basis, n_basis);
    // If I were to use a hybrid functional, I would need to build K and
    // access the percentage of HF exchange via the func methds for libxc
    // K = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int u = 0; u < n_basis; u++) {
      for (int v = 0; v <= u; v++) {
        f64 Juv = 0.0;
        for (int l = 0; l < n_basis; l++) {
          for (int s = 0; s <= l; s++) {
            double Pls = (l == s) ? P_new(l, s) : 2.0 * P_new(l, s);
            Juv += ERI[ten_idx(u, v, l, s)] * Pls;
            // K(u, v) += ERI[ten_idx(u, l, v, s, n_basis)] * P_new(l, s);
          }
        }
        J(u, v) = Juv;
        J(v, u) = Juv;
      }
    }
    double time_j = elapsed(t_j);

    P_old = P_new;
    E_old = E_new;

    auto t_xc = clk::now();
    spdlog::trace("Building rho (Iter {})...", (100 - max_iter));

    // I thought batching might help out with the memory and speed of the
    // building of rho, there's a paper by Stratmann, Scuseria, and Frisch
    // called "Achieving linear scaling n exchange-correlation density
    // functional quadratures" that I need to read and implement, but this
    // has worked well so far. Rho is what is passed to libxc to evaluate
    // the functional on
    int batch_size = 256;
    const f64 tol_rho = 1e-8;
    V_xc = Eigen::MatrixXd::Zero(n_basis, n_basis);
    rho = Eigen::VectorXd::Zero(total_points);
    exc = Eigen::VectorXd::Zero(total_points);
    vxc = Eigen::VectorXd::Zero(total_points);
    u64 n_screened = 0;

    for (int batch_start = 0; batch_start < total_points; batch_start += batch_size) {
      int actual_batch = std::min((int)total_points - batch_start, batch_size);

      auto chi_batch = chi.middleRows(batch_start, actual_batch);

      Eigen::VectorXd rho_batch = (chi_batch * P_old).cwiseProduct(chi_batch).rowwise().sum();

      std::vector<int> keep;
      keep.reserve(actual_batch);
      for (int i = 0; i < actual_batch; i++) {
        if (rho_batch(i) > tol_rho) {
          keep.push_back(i);
        }
      }
      n_screened += (actual_batch - keep.size());
      if (keep.empty()) {
        continue;
      }

      Eigen::MatrixXd chi_compact = chi_batch(keep, Eigen::placeholders::all);
      Eigen::VectorXd rho_compact = rho_batch(keep);
      Eigen::VectorXd w_compact = final_weights.segment(batch_start, actual_batch)(keep);

      Eigen::VectorXd exc_compact(keep.size()), vxc_compact(keep.size());
      xc_lda_exc(&func, keep.size(), rho_compact.data(), exc_compact.data());
      xc_lda_vxc(&func, keep.size(), rho_compact.data(), vxc_compact.data());

      std::vector<int> global_keep(keep.size());
      for (size_t i = 0; i < keep.size(); i++) {
        global_keep[i] = batch_start + keep[i];
      }

      rho(global_keep) = rho_compact;
      exc(global_keep) = exc_compact;

      Eigen::VectorXd w_vxc_compact = w_compact.array() * vxc_compact.array();
      V_xc.noalias() += chi_compact.transpose() * w_vxc_compact.asDiagonal() * chi_compact;
    }
    double time_xc = elapsed(t_xc);
    spdlog::trace("XC screened {}/{} points ({:.1f}%)", n_screened, total_points, 100.0 * n_screened / total_points);
    spdlog::trace("Building F matrix (Iter {})...", (100 - max_iter));

    // Now the rest of the code resembles HF, there's also a bit
    // of DIIS in here as well to help with smoothing
    F.noalias() = H + J + V_xc;

    // Start of DIIS
    spdlog::trace("Checking error (Iter {})...", (100 - max_iter));
    Eigen::MatrixXd e = F * P_old * S - S * P_old * F;
    spdlog::trace("Storing F matrix (Iter {})...", (100 - max_iter));
    fock_history.push_back(F);
    spdlog::trace("Storing error matrix (Iter {})...", (100 - max_iter));
    error_history.push_back(e);

    if (fock_history.size() > diis_size) {
      fock_history.pop_front();
      error_history.pop_front();
    }
    int n_diis = fock_history.size();

    spdlog::trace("Zeroing B matrix (Iter {})...", (100 - max_iter));
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_diis + 1, n_diis + 1);
    for (int i = 0; i < n_diis; i++) {
      for (int j = 0; j < n_diis; j++) {
        B(i, j) = (error_history[i].array() * error_history[j].array()).sum();
      }
    }
    B.col(n_diis).setConstant(-1.0);
    B.row(n_diis).setConstant(-1.0);
    B(n_diis, n_diis) = 0.0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_diis + 1);
    rhs(n_diis) = -1.0;
    Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

    spdlog::trace("Creating F_diis matrix (Iter {})...", (100 - max_iter));
    Eigen::MatrixXd F_diis = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_diis; i++) {
      F_diis += c(i) * fock_history[i];
    }
    // End of DIIS

    spdlog::trace("Solving F_diis matrix (Iter {})...", (100 - max_iter));
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(F_diis, S);
    Eigen::MatrixXd C = solver.eigenvectors();
    spdlog::trace("Creating C and P_new matrices (Iter {})...", (100 - max_iter));
    P_new = 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
    // So weirdly, I am not able to get great convergence without damping enabled, which is just
    // mixing a bit of the old density in with th enew density. NWChem doesn't seem to need to do
    // this, as evidenced by d=0.0 in the output, so I'm a little disappointed that I have to use
    // it just to get good convergence
    if (damping > 0.0) {
      P_new = (1.0 - damping) * P_new + damping * P_old;
    }

    spdlog::trace("Calculating E_new (Iter {})...", (100 - max_iter));
    E_new = (P_new.array() * (H + 0.5 * J).array()).sum() +
            (final_weights.array() * exc.array() * rho.array()).sum();
    spdlog::info("E_elec (Iter {}): {:.10f} dE: {:.10f}", (100 - max_iter), E_new, std::abs(E_new - E_old));
    spdlog::info("Iter {}: J={:.2f}s, XC={:.2f}s", (100 - max_iter), time_j, time_xc);

    if (std::abs(E_new - E_old) < conv_tol) {
      converged = true;
      spdlog::info("DFT SCF Converged!");
    }
  }
  // End fo the loop!
  f64 E_one = (P_new.array() * H.array()).sum();
  f64 E_coul = 0.5 * (P_new.array() * J.array()).sum();
  f64 E_xc = (final_weights.array() * exc.array() * rho.array()).sum();
  for (int i = 0; i < n_atoms; i++) {
    for (int j = i + 1; j < n_atoms; j++) {
      f64 R = (atom_pos.row(i) - atom_pos.row(j)).norm();
      E_nuc += atoms[i].atomic_number * atoms[j].atomic_number / R;
    }
  }

  spdlog::info("Total DFT Energy: {:.16f} Ha", (E_new + E_nuc));
  spdlog::info("One electron: {:.16f} Ha", E_one);
  spdlog::info("Coulomb: {:.16f} Ha", E_coul);
  spdlog::info("XC: {:.16f} Ha", E_xc);
  spdlog::info("Nuclear Repulson: {:.16f} Ha", E_nuc);
  spdlog::info("Total time: {:.6f}s", elapsed(t_main));
  return 0;
}
