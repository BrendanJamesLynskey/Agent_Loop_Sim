# Context module results (engine 1.5.0)

Written by `scripts/make_fixtures.py` from the shipped data (CI checks it is up to date). Every number on the Context site and in the README comes from here or from the same engine at build time.

Corpus: 6 articles of the SQuAD v1.1 development set (CC BY-SA 4.0), 200 questions (seed 23 from 1160). Embedder `sentence-transformers/all-MiniLM-L6-v2` @ `1110a24` (int8, per-vector scale); reranker `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d` (recursive-256 only).

## Chunking

| config | chunks | mean tokens | max tokens | questions lost (answer cut) | chunks over the embedder's 256 word pieces |
|---|---|---|---|---|---|
| fixed-128 | 296 | 126.7 | 128 | 5 | 0 |
| fixed-256 | 150 | 250.0 | 256 | 4 | 55 |
| fixed-512 | 77 | 487.0 | 512 | 1 | 73 |
| fixed-256-o64 | 197 | 252.3 | 256 | 0 | 75 |
| recursive-256 | 208 | 180.3 | 256 | 0 | 7 |
| semantic-256 | 351 | 106.8 | 256 | 1 | 9 |

## Retrieval (means over all questions; a question with its answer cut by chunking scores 0)

| config | method | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|---|
| fixed-128 | bm25 | 0.680 | 0.890 | 0.920 | 0.945 | 0.950 | 0.786 | 0.826 |
| fixed-128 | dense precision=int8 | 0.555 | 0.810 | 0.855 | 0.900 | 0.950 | 0.685 | 0.738 |
| fixed-128 | dense precision=int4 | 0.565 | 0.790 | 0.855 | 0.900 | 0.955 | 0.687 | 0.740 |
| fixed-128 | dense precision=binary | 0.475 | 0.705 | 0.790 | 0.875 | 0.925 | 0.611 | 0.675 |
| fixed-128 | rrf | 0.700 | 0.880 | 0.930 | 0.945 | 0.970 | 0.796 | 0.833 |
| fixed-128 | weighted alpha=0.3 | 0.725 | 0.915 | 0.940 | 0.965 | 0.970 | 0.821 | 0.857 |
| fixed-128 | weighted alpha=0.5 | 0.745 | 0.910 | 0.955 | 0.965 | 0.970 | 0.833 | 0.866 |
| fixed-128 | weighted alpha=0.7 | 0.730 | 0.880 | 0.945 | 0.955 | 0.970 | 0.816 | 0.851 |
| fixed-256 | bm25 | 0.730 | 0.930 | 0.955 | 0.970 | 0.980 | 0.829 | 0.865 |
| fixed-256 | dense precision=int8 | 0.595 | 0.805 | 0.900 | 0.945 | 0.975 | 0.718 | 0.773 |
| fixed-256 | dense precision=int4 | 0.615 | 0.810 | 0.895 | 0.950 | 0.975 | 0.727 | 0.781 |
| fixed-256 | dense precision=binary | 0.470 | 0.685 | 0.800 | 0.930 | 0.965 | 0.604 | 0.682 |
| fixed-256 | rrf | 0.735 | 0.940 | 0.970 | 0.975 | 0.980 | 0.837 | 0.872 |
| fixed-256 | weighted alpha=0.3 | 0.795 | 0.955 | 0.970 | 0.980 | 0.980 | 0.873 | 0.900 |
| fixed-256 | weighted alpha=0.5 | 0.805 | 0.960 | 0.970 | 0.980 | 0.980 | 0.877 | 0.903 |
| fixed-256 | weighted alpha=0.7 | 0.775 | 0.940 | 0.975 | 0.980 | 0.980 | 0.854 | 0.885 |
| fixed-512 | bm25 | 0.745 | 0.945 | 0.975 | 0.990 | 0.995 | 0.841 | 0.879 |
| fixed-512 | dense precision=int8 | 0.475 | 0.665 | 0.755 | 0.885 | 0.980 | 0.597 | 0.665 |
| fixed-512 | dense precision=int4 | 0.475 | 0.670 | 0.765 | 0.870 | 0.975 | 0.597 | 0.662 |
| fixed-512 | dense precision=binary | 0.380 | 0.600 | 0.710 | 0.850 | 0.955 | 0.522 | 0.600 |
| fixed-512 | rrf | 0.635 | 0.820 | 0.940 | 0.990 | 0.990 | 0.754 | 0.812 |
| fixed-512 | weighted alpha=0.3 | 0.755 | 0.955 | 0.985 | 0.995 | 0.995 | 0.855 | 0.890 |
| fixed-512 | weighted alpha=0.5 | 0.765 | 0.940 | 0.975 | 0.995 | 0.995 | 0.856 | 0.891 |
| fixed-512 | weighted alpha=0.7 | 0.645 | 0.865 | 0.920 | 0.975 | 0.995 | 0.759 | 0.812 |
| fixed-256-o64 | bm25 | 0.647 | 0.890 | 0.943 | 0.983 | 0.990 | 0.854 | 0.876 |
| fixed-256-o64 | dense precision=int8 | 0.438 | 0.738 | 0.818 | 0.920 | 0.993 | 0.689 | 0.730 |
| fixed-256-o64 | dense precision=int4 | 0.448 | 0.743 | 0.813 | 0.912 | 0.993 | 0.692 | 0.728 |
| fixed-256-o64 | dense precision=binary | 0.380 | 0.662 | 0.777 | 0.875 | 0.973 | 0.619 | 0.663 |
| fixed-256-o64 | rrf | 0.565 | 0.885 | 0.958 | 0.990 | 0.998 | 0.813 | 0.851 |
| fixed-256-o64 | weighted alpha=0.3 | 0.667 | 0.922 | 0.960 | 0.990 | 0.998 | 0.876 | 0.900 |
| fixed-256-o64 | weighted alpha=0.5 | 0.655 | 0.925 | 0.963 | 0.988 | 0.998 | 0.872 | 0.895 |
| fixed-256-o64 | weighted alpha=0.7 | 0.598 | 0.890 | 0.963 | 0.983 | 0.998 | 0.831 | 0.861 |
| recursive-256 | bm25 | 0.785 | 0.950 | 0.985 | 0.990 | 0.995 | 0.869 | 0.899 |
| recursive-256 | dense precision=int8 | 0.705 | 0.885 | 0.950 | 0.980 | 0.995 | 0.806 | 0.849 |
| recursive-256 | dense precision=int4 | 0.700 | 0.895 | 0.935 | 0.980 | 0.995 | 0.804 | 0.848 |
| recursive-256 | dense precision=binary | 0.600 | 0.805 | 0.870 | 0.960 | 0.985 | 0.720 | 0.778 |
| recursive-256 | rrf | 0.810 | 0.965 | 0.980 | 1.000 | 1.000 | 0.883 | 0.912 |
| recursive-256 | weighted alpha=0.3 | 0.820 | 0.970 | 0.985 | 0.995 | 1.000 | 0.893 | 0.919 |
| recursive-256 | weighted alpha=0.5 | 0.840 | 0.975 | 0.990 | 0.995 | 1.000 | 0.906 | 0.928 |
| recursive-256 | weighted alpha=0.7 | 0.830 | 0.975 | 0.990 | 1.000 | 1.000 | 0.901 | 0.926 |
| semantic-256 | bm25 | 0.710 | 0.895 | 0.950 | 0.970 | 0.975 | 0.810 | 0.850 |
| semantic-256 | dense precision=int8 | 0.605 | 0.835 | 0.900 | 0.955 | 0.965 | 0.732 | 0.786 |
| semantic-256 | dense precision=int4 | 0.610 | 0.840 | 0.900 | 0.955 | 0.975 | 0.732 | 0.787 |
| semantic-256 | dense precision=binary | 0.535 | 0.765 | 0.845 | 0.915 | 0.950 | 0.667 | 0.727 |
| semantic-256 | rrf | 0.735 | 0.935 | 0.960 | 0.980 | 0.990 | 0.832 | 0.869 |
| semantic-256 | weighted alpha=0.3 | 0.760 | 0.940 | 0.960 | 0.975 | 0.985 | 0.847 | 0.879 |
| semantic-256 | weighted alpha=0.5 | 0.755 | 0.940 | 0.970 | 0.980 | 0.985 | 0.848 | 0.882 |
| semantic-256 | weighted alpha=0.7 | 0.750 | 0.925 | 0.955 | 0.980 | 0.990 | 0.839 | 0.874 |
| recursive-256 | rerank n=10 | 0.890 | 0.990 | 0.995 | 1.000 | 1.000 | 0.938 | 0.954 |
| recursive-256 | rerank n=20 | 0.890 | 0.990 | 0.995 | 0.995 | 1.000 | 0.937 | 0.952 |
| recursive-256 | rerank n=30 | 0.890 | 0.990 | 0.995 | 0.995 | 1.000 | 0.937 | 0.952 |
| recursive-256 | rerank base=bm25 n=10 | 0.885 | 0.980 | 0.985 | 0.990 | 0.995 | 0.930 | 0.946 |
| recursive-256 | rerank base=dense n=10 | 0.890 | 0.975 | 0.975 | 0.980 | 0.995 | 0.930 | 0.942 |
| recursive-256 | bm25 k1=0.5 b=0.75 | 0.785 | 0.945 | 0.975 | 0.990 | 0.995 | 0.870 | 0.901 |
| recursive-256 | bm25 k1=2.0 b=0.75 | 0.775 | 0.950 | 0.985 | 0.990 | 0.995 | 0.864 | 0.896 |
| recursive-256 | bm25 k1=1.2 b=0.0 | 0.790 | 0.940 | 0.975 | 0.990 | 0.995 | 0.871 | 0.901 |
| recursive-256 | bm25 k1=1.2 b=1.0 | 0.780 | 0.960 | 0.980 | 0.990 | 0.995 | 0.867 | 0.898 |
| recursive-256 | rrf k=10 | 0.810 | 0.970 | 0.990 | 0.995 | 1.000 | 0.886 | 0.913 |
| recursive-256 | rrf depth=10 | 0.810 | 0.955 | 0.985 | 0.995 | 1.000 | 0.882 | 0.911 |

## Offline float32 dense retrieval (recorded by scripts/build_context_data.py; not reproducible from the shipped int8 vectors)

| config | recall@1 | recall@3 | recall@5 | recall@10 | recall@20 | mrr@10 | ndcg@10 |
|---|---|---|---|---|---|---|---|
| fixed-128 | 0.550 | 0.810 | 0.855 | 0.895 | 0.950 | 0.682 | 0.734 |
| fixed-256 | 0.595 | 0.800 | 0.900 | 0.945 | 0.975 | 0.718 | 0.773 |
| fixed-512 | 0.475 | 0.665 | 0.755 | 0.885 | 0.980 | 0.597 | 0.665 |
| fixed-256-o64 | 0.432 | 0.740 | 0.818 | 0.922 | 0.993 | 0.687 | 0.730 |
| recursive-256 | 0.705 | 0.885 | 0.950 | 0.980 | 0.995 | 0.806 | 0.849 |
| semantic-256 | 0.605 | 0.835 | 0.900 | 0.955 | 0.965 | 0.732 | 0.786 |

## The window task (12 questions, BM25 over recursive-256, up to 3 reads each)

| budget | policy | facts answered | input tokens spent | model calls | reads | compactions | reads dropped | peak window |
|---|---|---|---|---|---|---|---|---|
| 1000 | unbounded | 12 of 12 | 26616 | 16 | 15 | 0 | 0 | 3033 |
| 1000 | truncate | 3 of 12 | 12867 | 16 | 15 | 0 | 11 | 1237 |
| 1000 | compact | 12 of 12 | 15149 | 20 | 15 | 4 | 0 | 1237 |
| 1000 | retrieve | 12 of 12 | 26561 | 31 | 30 | 0 | 26 | 1237 |
| 1000 | compact+retrieve | 12 of 12 | 15149 | 20 | 15 | 4 | 0 | 1237 |
| 1500 | unbounded | 12 of 12 | 26616 | 16 | 15 | 0 | 0 | 3033 |
| 1500 | truncate | 5 of 12 | 18094 | 16 | 15 | 0 | 9 | 1640 |
| 1500 | compact | 12 of 12 | 16768 | 18 | 15 | 2 | 0 | 1652 |
| 1500 | retrieve | 12 of 12 | 39175 | 31 | 30 | 0 | 24 | 1678 |
| 1500 | compact+retrieve | 12 of 12 | 16768 | 18 | 15 | 2 | 0 | 1652 |
| 2000 | unbounded | 12 of 12 | 26616 | 16 | 15 | 0 | 0 | 3033 |
| 2000 | truncate | 8 of 12 | 21803 | 16 | 15 | 0 | 6 | 2136 |
| 2000 | compact | 12 of 12 | 19422 | 17 | 15 | 1 | 0 | 2136 |
| 2000 | retrieve | 12 of 12 | 49919 | 31 | 30 | 0 | 21 | 2191 |
| 2000 | compact+retrieve | 12 of 12 | 19422 | 17 | 15 | 1 | 0 | 2136 |
| 3000 | unbounded | 12 of 12 | 26616 | 16 | 15 | 0 | 0 | 3033 |
| 3000 | truncate | 12 of 12 | 26486 | 16 | 15 | 0 | 1 | 3033 |
| 3000 | compact | 12 of 12 | 27223 | 17 | 15 | 1 | 0 | 3033 |
| 3000 | retrieve | 12 of 12 | 26486 | 16 | 15 | 0 | 1 | 3033 |
| 3000 | compact+retrieve | 12 of 12 | 27223 | 17 | 15 | 1 | 0 | 3033 |
