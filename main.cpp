#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::seq;

MatrixXd read_tableau(const std::string& filePath) {
  std::ifstream file(filePath);
  if(!file.is_open())
    throw std::ifstream::failure("File could not be opened");
  int m, n;
  file >> m >> n;
  MatrixXd tableau(m+1, n+1);
  for (int i = 0; i < m + 1; i++)
      for (int j = 0; j < n + 1; j++)
        file >> tableau(i, j);
  file.close();
  return tableau;
}

auto initialize(int m, int n) {
  VectorXd x(n), y(m), s(n);
  double mu;
  std::random_device r;
  std::default_random_engine engine(r());
  std::uniform_real_distribution<> u_dist(-1.0, 1.0);
  double positive_lower = std::nextafter(0, std::numeric_limits<double>::max());
  // Positive uniform distribution
  std::uniform_real_distribution<> u_dist_pos(positive_lower, 1.0);
  for(int i = 0; i < n; i++) {
    x(i) = u_dist_pos(engine);
    s(i) = u_dist_pos(engine);
  }
  for (int i = 0; i < m; i++)
    y(i) = u_dist(engine);
  mu = u_dist_pos(engine);
  return std::make_tuple(x, y, s, mu);
}

double get_alpha(const VectorXd& vec, const VectorXd& delta_vec) {
  assert(vec.size() == delta_vec.size());
  int size = vec.size();
  double alpha = std::numeric_limits<double>::max();
  bool all_delta_nonnegative= true;
  double ratio;
  for(int i = 0; i < size; i++)
    if(delta_vec(i) < 0 && (ratio = -(vec(i) / delta_vec(i))) < alpha) {
      alpha = ratio;
      all_delta_nonnegative = false;
    }
  return all_delta_nonnegative ? 1.0 : alpha;
}

auto interior_point(const MatrixXd& A, const VectorXd& b, const VectorXd& c,
                    double eps = 10e-6, double max_norm = 10e6,
                    int k_max = 1000) {
  int m = A.rows(), n = A.cols();
  assert(m == b.size());
  assert(n == c.size());
  double sq_norm, max_sq_norm = std::pow(max_norm, 2);
  constexpr double fraction = 1 - 10e-7;
  double theta = (n > 13) ? (1 - 3.5 / std::sqrt(n)) : 0.5;
  auto [x, y, s, mu] = initialize(m, n);
  int k = 0;
  do {
    MatrixXd S = s.asDiagonal();
    MatrixXd D = (x.array() / s.array()).matrix().asDiagonal();
    VectorXd rho_P = b - A * x;
    VectorXd rho_D = c - A.transpose() * y - s;
    VectorXd v = (mu - (x.array() * s.array())).matrix().transpose();
    VectorXd delta_y = -((A * D * A.transpose()).inverse() *
                         (A * S.inverse() * v - A * D * rho_D - rho_P));
    VectorXd delta_s = -(A.transpose() * delta_y) + rho_D;
    VectorXd delta_x = S.inverse() * v - D * delta_s;
    double alpha = fraction * std::min(get_alpha(x, delta_x),
                                       get_alpha(s, delta_s));
    VectorXd x_new = x + alpha * delta_x;
    VectorXd y_new = y + alpha * delta_y;
    VectorXd s_new = s + alpha * delta_s;

    // Concatenate vectors.
    VectorXd joined(x.size() + y.size() + s.size());
    joined << x, y, s;
    VectorXd joined_new(x_new.size() + y_new.size() + s_new.size());
    joined_new << x_new, y_new, s_new;
    sq_norm = (joined_new - joined).squaredNorm();
    // sq_norm = std::max({
    //     (x_new - x).squaredNorm(),
    //     (y_new - y).squaredNorm(),
    //     (s_new - s).squaredNorm()
    //   });
    x = x_new;
    y = y_new;
    s = s_new;
    mu *= theta;
    k++;
  } while((x.transpose() * s > eps) &&
          (k < k_max) &&
          (sq_norm < max_sq_norm));
  if(x.transpose() * s < eps)
    return std::make_tuple(x, y);
  else if(sq_norm > max_sq_norm)
    throw std::invalid_argument("Algorithm did not converge. Norm exceeded.");
  else
    throw std::invalid_argument("Algorithm did not converge."
                                "Number of maximum iterations exceeded.");
}

int main(int argc, char** argv) {
  // Process command-line arguments
  std::string file_path = "";
  std::vector<std::string> args(argv + 1, argv + argc);
  for(auto arg = args.begin(); arg != args.end(); arg++) {
    if(*arg == "-i")
      file_path = *(++arg);
  }
  if(file_path.empty()) {
    std::cout << "USAGE: pdip -i <file>" << std::endl;
    return EXIT_FAILURE;
  }

  MatrixXd tableau;
  try {
    tableau = read_tableau(file_path);
  }
  catch(std::ifstream::failure& ex) {
    std::cout << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  auto start = std::chrono::steady_clock::now();

  int m = tableau.rows() - 1, n = tableau.cols() - 1;
  MatrixXd A = tableau(seq(0, m-1), seq(0, n-1));
  VectorXd b = tableau(seq(0, m-1), n);
  VectorXd c = tableau(m, seq(0, n-1));

  VectorXd x_sum = VectorXd::Zero(n);
  VectorXd y_sum = VectorXd::Zero(m);
  int n_runs = 30;
  for (int i = 0; i < n_runs; i++) {
    auto [x, y] = interior_point(A, b, c, 10e-10);
    x_sum += x;
    y_sum += y;
  }

  std::cout << "Average solutions after " << n_runs << " executions:"
            << std::endl;
  VectorXd x_avg = x_sum.transpose().array() / n_runs;
  VectorXd y_avg = y_sum.transpose().array() / n_runs;

  auto end = std::chrono::steady_clock::now();

  std::cout << "x:" << std::endl << x_avg << std::endl;
  std::cout << std::endl;
  std::cout << "y:" << std::endl << y_avg << std::endl;
  std::cout << std::endl;
  std::cout << "optimum:" << std::endl << x_avg.dot(c) << std::endl;
  std::cout << std::endl;
  std::cout << "Running time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / n_runs
            << '\n';
  return EXIT_SUCCESS;
}
