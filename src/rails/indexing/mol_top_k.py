# Retrieval with Learned Similarities (RAILS, https://arxiv.org/abs/2407.15462).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines exact- and approximate- Top-K modules for Mixture-of-Logits (MoL),
discussed in Retrieval with Learned Similarities (https://arxiv.org/abs/2407.15462).
"""

from typing import Dict, Tuple

import torch
from torch.profiler import record_function

from src.rails.indexing.candidate_index import TopKModule
from src.rails.similarities.mol.similarity_fn import MoLSimilarity


class MoLTopKModule(TopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        flatten_item_ids_and_embeddings: bool,
        keep_component_level_item_embeddings: bool,
        component_level_item_embeddings_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, X, D) if mol_module._apply_item_embeddings_fn is True,
                (1, X, P_X, D_P) otherwise.
            item_ids: (1, X,) representing the item ids.
            flatten_item_ids_and_embeddings: bool. If true, do not keep the extra (1,)
                dimension at size(0).
            keep_component_level_item_embeddings: bool. If true, keep P_x component-level
                embeddings in `self._mol_item_embeddings` for downstream applications.
            component_level_item_embeddings_dtype: torch.dtype. If set, the dtype
                to keep component-level item embeddings in. By default we use bfloat16.
        """
        super().__init__()

        self._mol_module: MoLSimilarity = mol_module
        self._item_embeddings: torch.Tensor = (
            item_embeddings
            if not flatten_item_ids_and_embeddings
            else item_embeddings.squeeze(0)
        )

        if keep_component_level_item_embeddings:
            self._mol_item_embeddings: torch.Tensor = (
                mol_module.get_item_component_embeddings(
                    (
                        self._item_embeddings.squeeze(0)
                        if not flatten_item_ids_and_embeddings
                        else self._item_embeddings
                    ),
                    decoupled_inference=True,
                )[
                    0
                ]  # (X, D) -> (X, P_X, D_P)
            ).to(component_level_item_embeddings_dtype)

        self._item_ids: torch.Tensor = (
            item_ids if not flatten_item_ids_and_embeddings else item_ids.squeeze(0)
        )

    @property
    def mol_module(self) -> MoLSimilarity:
        return self._mol_module


class MoLBruteForceTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=False,
            keep_component_level_item_embeddings=False,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, D) if mol_module._apply_query_embeddings_fn is True,
                (B, X, P_Q, D_P) otherwise.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
            **kwargs: Implementation-specific keys/values.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits, _ = self.mol_module(
            query_embeddings,
            self._item_embeddings,
            **kwargs,
        )
        top_k_logits, top_k_indices = torch.topk(
            all_logits,
            dim=1,
            k=k,
            sorted=sorted,
            largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]


class MoLNaiveTopK(MoLTopKModule):
    """
    The naive top 𝐾 algorithm is a greedy based algorithm to retrieve the top 𝐾 items. The key idea
    is to leverage the dot product scores to scope the set of items for calculating the
    learned similarity scores.

    The algorithm works as follows:
    • Retrieve the top 𝐾 for each set of embeddings by their dot similarity scores;
    • Takes the union of the retrieved items as 𝐼;
    • Retrieve the dot similarity scores of all the items in 𝐼 for each embedding set;
    • Calculate the learned similarity score for all the items in 𝐼;
    • Returns the top 𝐾 items from 𝐼 with the highest learned similarity score.
    """

    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        k_per_group: int,
        use_faiss: bool = False,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, X, D).
            item_ids: (1, X,).
            k_per_group: int. top-k (k_per_group) per group to take union.
            use_faiss: bool. If true, use FAISS to accelerate the top-k search.
        """
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        N, P_X, D_P = self._mol_item_embeddings.size()
        self._k_per_group: int = k_per_group
        self._mol_item_embeddings_t: torch.Tensor = (
            self._mol_item_embeddings.permute(1, 0, 2).reshape(-1, D_P).transpose(0, 1)
        )  # (N, P_X, D_P) -> (P_X, N, D_P) -> (P_X * N, D_P) -> (D_P, P_X * N)
        self._use_faiss: bool = use_faiss
        if use_faiss:
            import faiss

            self._gpu_resources = faiss.StandardGpuResources()
            self._gpu_indexes = []
            nlist = 100
            for i in range(P_X):
                mol_item_embeddings_np = (
                    self._mol_item_embeddings[:, i, :]
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .astype("float16")
                )
                quantizer = faiss.IndexFlatIP(D_P)
                index = faiss.IndexIVFFlat(
                    quantizer, D_P, nlist, faiss.METRIC_INNER_PRODUCT
                )
                assert (
                    not index.is_trained
                )  # make sure the index is not already trained
                index.train(mol_item_embeddings_np)  # train with the dataset vectors
                assert index.is_trained  # verify the index is now trained
                gpu_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, index)
                gpu_index.add(mol_item_embeddings_np)
                self._gpu_indexes.append(gpu_index)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, D) if mol_module._apply_query_embeddings_fn is True,
                (B, X, P_Q, D_P) otherwise.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
            **kwargs: Implementation-specific keys/values.
        """
        B: int = query_embeddings.size(0)
        mol_query_embeddings, _ = self.mol_module.get_query_component_embeddings(
            query_embeddings,
            decoupled_inference=True,
            **kwargs,
        )  # (B, P_Q, D)
        _, P_Q, _ = mol_query_embeddings.size()
        N, P_X, _ = self._mol_item_embeddings.size()
        all_indices = []
        if self._use_faiss:
            for i in range(P_Q):
                for j in range(P_X):
                    _, ij_indices = self._gpu_indexes[j].search(
                        mol_query_embeddings[:, i, :].cpu().to(torch.float32).numpy(),
                        self._k_per_group,
                    )
                    all_indices.append(
                        torch.tensor(
                            ij_indices,
                            dtype=torch.int64,
                            device=query_embeddings.device,
                        )
                    )
        else:
            mol_query_component_embeddings = torch.tensor_split(
                mol_query_embeddings.to(self._mol_item_embeddings_t.dtype),
                P_Q,
                dim=1,
            )
            for i in range(P_Q):
                cur_i_sim_values = torch.mm(
                    mol_query_component_embeddings[i].squeeze(1),
                    self._mol_item_embeddings_t,
                ).view(B * P_X, N)
                _, cur_i_top_k_indices = torch.topk(
                    cur_i_sim_values, k=self._k_per_group, dim=1, sorted=False
                )
                all_indices.append(cur_i_top_k_indices.view(B, P_X * self._k_per_group))

        sorted_all_indices, _ = torch.sort(torch.cat(all_indices, dim=1), dim=1)

        # MoL
        k = P_Q * P_X * self._k_per_group
        filtered_item_embeddings = self._item_embeddings[
            sorted_all_indices.view(-1)
        ].reshape(
            (
                B,
                k,
            )
            + self._item_embeddings.size()[1:]
        )
        candidate_scores, _ = self.mol_module(
            query_embeddings,
            filtered_item_embeddings,
            **kwargs,
        )
        # (B, P_Q * P_X * self._k_per_group).
        # Mask out duplicate elements across multiple top-k groups, given input is sorted.
        candidate_is_valid = torch.cat(
            [
                torch.ones_like(sorted_all_indices[:, 0:1], dtype=torch.bool),
                sorted_all_indices[:, 1:] != sorted_all_indices[:, :-1],
            ],
            dim=1,
        )
        candidate_scores = torch.where(candidate_is_valid, candidate_scores, -32767.0)
        top_k_logits, top_k_indices = torch.topk(
            input=candidate_scores, k=k, dim=1, largest=True, sorted=sorted
        )
        return (
            top_k_logits,
            self._item_ids[
                torch.gather(sorted_all_indices, dim=1, index=top_k_indices)
            ],
        )


class MoLAvgTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, N, D).
            item_ids: (1, N,).
            avg_top_k: int.
        """
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        _, P_X, D_P = self._mol_item_embeddings.size()
        self._P_X: int = P_X
        self._D_P: int = D_P
        self._avg_top_k: int = avg_top_k
        self.register_buffer(
            "_avg_mol_item_embeddings_t",
            self._mol_item_embeddings.mean(1).transpose(0, 1),
        )  # (N, D_P) -> (D_P, N)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, D) if mol_module._apply_query_embeddings_fn is True,
                (B, X, P_Q, D_P) otherwise.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
            **kwargs: Implementation-specific keys/values.
        """
        B: int = query_embeddings.size(0)
        avg_mol_query_embeddings = self.mol_module.get_query_component_embeddings(
            query_embeddings,
            decoupled_inference=True,
            **kwargs,
        )[0].mean(
            1
        )  # [B, P_Q, D_P] -> [B, D_P]

        with record_function("avg_top_k_scoring"):
            avg_sim_values = torch.mm(
                avg_mol_query_embeddings.to(self._avg_mol_item_embeddings_t.dtype),
                self._avg_mol_item_embeddings_t,
            )  # [B, D_P], [D_P, N] -> [B, N]
            _, avg_sim_top_k_indices = torch.topk(
                avg_sim_values, dim=1, k=self._avg_top_k, sorted=False
            )  # [B, avg_top_k]

        with record_function("avg_top_k_gather_embeddings"):
            filtered_item_embeddings = self._item_embeddings[
                avg_sim_top_k_indices.view(-1)
            ].reshape((B, self._avg_top_k) + self._item_embeddings.size()[1:])

        # MoL scoring and final top-k.
        with record_function("avg_top_k_mol_scoring"):
            candidate_scores, _ = self.mol_module(
                query_embeddings,
                filtered_item_embeddings,
                **kwargs,
            )
        final_k = min(k, candidate_scores.size(1))
        top_k_logits, top_k_indices = torch.topk(
            candidate_scores,
            dim=1,
            k=final_k,
            largest=True,
            sorted=sorted,
        )  # [B, k]
        return top_k_logits, self._item_ids[
            torch.gather(avg_sim_top_k_indices, dim=1, index=top_k_indices)
        ]


class MoLCombTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        avg_top_k: int,
        k_per_group: int,
    ) -> None:
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=True,
            keep_component_level_item_embeddings=True,
        )
        self._avg_top_k_module = MoLAvgTopK(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            avg_top_k=avg_top_k,
        )
        self._naive_top_k_module = MoLNaiveTopK(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            k_per_group=k_per_group,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        avg_top_k_logits, avg_top_k_ids = self._avg_top_k_module(
            query_embeddings=query_embeddings,
            k=k,
            sorted=sorted,
            **kwargs,
        )
        naive_top_k_logits, naive_top_k_ids = self._naive_top_k_module(
            query_embeddings=query_embeddings,
            k=k,
            sorted=sorted,
            **kwargs,
        )
        top_k_logits = torch.cat([avg_top_k_logits, naive_top_k_logits], dim=1)
        top_k_ids = torch.cat([avg_top_k_ids, naive_top_k_ids], dim=1)
        final_k = min(k, top_k_logits.size(1))
        top_k_logits, top_k_indices = torch.topk(
            top_k_logits,
            dim=1,
            k=final_k,
            sorted=sorted,
            largest=True,
        )
        return top_k_logits, torch.gather(top_k_ids, dim=1, index=top_k_indices)
