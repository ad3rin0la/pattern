#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

template <typename scalar_t>
double squared_distance(
    const scalar_t* lhs, const scalar_t* rhs, std::int64_t lhs_index,
    std::int64_t rhs_index) {
  double total = 0.0;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    const double delta = static_cast<double>(lhs[3 * lhs_index + axis]) -
                         static_cast<double>(rhs[3 * rhs_index + axis]);
    total += delta * delta;
  }
  return total;
}

template <typename scalar_t>
std::vector<torch::Tensor> traverse(
    const torch::Tensor& queries, const std::vector<torch::Tensor>& centers,
    const std::vector<torch::Tensor>& incidences, std::int64_t top_coarse,
    std::int64_t max_children) {
  const auto n_queries = queries.size(0);
  const auto n_ranks = static_cast<std::int64_t>(centers.size());
  std::vector<torch::Tensor> result(n_ranks);
  const scalar_t* query_data = queries.data_ptr<scalar_t>();

  const auto coarse_count = centers.back().size(0);
  const auto coarse_width = std::min(top_coarse, coarse_count);
  auto coarse = torch::empty({n_queries, coarse_width}, torch::TensorOptions().dtype(torch::kLong));
  auto* coarse_out = coarse.data_ptr<std::int64_t>();
  const scalar_t* coarse_centers = centers.back().data_ptr<scalar_t>();
  for (std::int64_t query = 0; query < n_queries; ++query) {
    std::vector<std::pair<double, std::int64_t>> ranked;
    ranked.reserve(coarse_count);
    for (std::int64_t cell = 0; cell < coarse_count; ++cell) {
      ranked.emplace_back(
          squared_distance(query_data, coarse_centers, query, cell), cell);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + coarse_width, ranked.end());
    for (std::int64_t slot = 0; slot < coarse_width; ++slot) {
      coarse_out[query * coarse_width + slot] = ranked[slot].second;
    }
  }
  result.back() = coarse;

  for (std::int64_t rank = n_ranks - 2; rank >= 0; --rank) {
    const auto& incidence = incidences[rank];
    const auto* incidence_data = incidence.data_ptr<std::int64_t>();
    const auto n_edges = incidence.size(1);
    const auto n_children = centers[rank].size(0);
    const auto n_parents = centers[rank + 1].size(0);
    const scalar_t* child_centers = centers[rank].data_ptr<scalar_t>();
    const auto& parent_candidates = result[rank + 1];
    const auto* parent_data = parent_candidates.data_ptr<std::int64_t>();
    const auto parent_width = parent_candidates.size(1);
    std::vector<std::vector<std::int64_t>> retained(n_queries);
    std::int64_t output_width = 1;

    // Build parent -> child CSR once per rank, then reuse it for every query.
    std::vector<std::int64_t> offsets(n_parents + 1, 0);
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      const auto child = incidence_data[edge];
      const auto parent = incidence_data[n_edges + edge];
      TORCH_CHECK(child >= 0 && child < n_children,
                  "incidence contains an out-of-range child index");
      TORCH_CHECK(parent >= 0 && parent < n_parents,
                  "incidence contains an out-of-range parent index");
      ++offsets[parent + 1];
    }
    for (std::int64_t parent = 0; parent < n_parents; ++parent) {
      offsets[parent + 1] += offsets[parent];
    }
    std::vector<std::int64_t> cursor(offsets.begin(), offsets.end() - 1);
    std::vector<std::int64_t> adjacency(n_edges);
    for (std::int64_t edge = 0; edge < n_edges; ++edge) {
      const auto child = incidence_data[edge];
      const auto parent = incidence_data[n_edges + edge];
      adjacency[cursor[parent]++] = child;
    }

    for (std::int64_t query = 0; query < n_queries; ++query) {
      std::unordered_set<std::int64_t> child_set;
      for (std::int64_t slot = 0; slot < parent_width; ++slot) {
        const auto parent = parent_data[query * parent_width + slot];
        if (parent < 0) continue;
        TORCH_CHECK(parent < n_parents, "candidate contains an invalid parent index");
        for (auto offset = offsets[parent]; offset < offsets[parent + 1]; ++offset) {
          child_set.insert(adjacency[offset]);
        }
      }
      if (child_set.empty()) {
        for (std::int64_t child = 0; child < n_children; ++child) child_set.insert(child);
      }
      std::vector<std::pair<double, std::int64_t>> ranked;
      ranked.reserve(child_set.size());
      for (const auto child : child_set) {
        ranked.emplace_back(
            squared_distance(query_data, child_centers, query, child), child);
      }
      const auto keep = std::min<std::int64_t>(max_children, ranked.size());
      std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end());
      retained[query].reserve(keep);
      for (std::int64_t slot = 0; slot < keep; ++slot) {
        retained[query].push_back(ranked[slot].second);
      }
      output_width = std::max<std::int64_t>(output_width, keep);
    }
    auto output = torch::full({n_queries, output_width}, -1, torch::TensorOptions().dtype(torch::kLong));
    auto* output_data = output.data_ptr<std::int64_t>();
    for (std::int64_t query = 0; query < n_queries; ++query) {
      for (std::int64_t slot = 0; slot < static_cast<std::int64_t>(retained[query].size()); ++slot) {
        output_data[query * output_width + slot] = retained[query][slot];
      }
    }
    result[rank] = output;
  }
  return result;
}

}  // namespace

std::vector<torch::Tensor> hierarchical_candidates(
    torch::Tensor queries, std::vector<torch::Tensor> centers,
    std::vector<torch::Tensor> incidences, std::int64_t top_coarse,
    std::int64_t max_children) {
  TORCH_CHECK(queries.device().is_cpu(), "queries must be a CPU tensor");
  TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3, "queries must have shape (Q, 3)");
  TORCH_CHECK(queries.is_floating_point(), "queries must be floating point");
  TORCH_CHECK(!centers.empty(), "centers cannot be empty");
  TORCH_CHECK(incidences.size() + 1 == centers.size(), "incidences must connect adjacent ranks");
  TORCH_CHECK(top_coarse > 0 && max_children > 0, "candidate limits must be positive");
  for (const auto& value : centers) {
    TORCH_CHECK(value.device().is_cpu(), "centers must be CPU tensors");
    TORCH_CHECK(value.scalar_type() == queries.scalar_type(), "centers must match query dtype");
    TORCH_CHECK(value.dim() == 2 && value.size(1) == 3, "centers must have shape (C, 3)");
  }
  for (const auto& value : incidences) {
    TORCH_CHECK(value.device().is_cpu() && value.scalar_type() == torch::kLong,
                "incidences must be CPU int64 tensors");
    TORCH_CHECK(value.dim() == 2 && value.size(0) == 2,
                "incidences must have shape (2, E)");
  }
  std::vector<torch::Tensor> output;
  AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "hierarchical_candidates", [&] {
    output = traverse<scalar_t>(queries, centers, incidences, top_coarse, max_children);
  });
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("hierarchical_candidates", &hierarchical_candidates,
             "CSR-style hierarchical electronic candidate traversal");
}
