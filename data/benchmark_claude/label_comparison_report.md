# gpt-5-mini vs claude-sonnet-4-6 — label comparison

Generated: 2026-05-03T16:43:50.084914

Same 800 reviewer claims, same 84 papers, same prompts. Only the labeling model changed.

## B2 (claim vs author response)

- N = 800
- Exact agreement: **480/800 (60.0%)**

### Label distribution
| label | gpt-5-mini | claude-sonnet-4-6 |
|---|---:|---:|
| Supported | 286 | 158 |
| Partially Supported | 299 | 390 |
| Contradicted | 148 | 163 |
| Not Determined | 67 | 89 |

### Confusion matrix (rows = gpt-5-mini, cols = claude-sonnet-4-6)
| gpt5 \ claude | Supported | Partially Supported | Contradicted | Not Determined | total |
|---|---|---|---|---|---|
| **Supported** | 126 | 107 | 42 | 11 | 286 |
| **Partially Supported** | 23 | 220 | 34 | 22 | 299 |
| **Contradicted** | 4 | 58 | 82 | 4 | 148 |
| **Not Determined** | 5 | 5 | 5 | 52 | 67 |

### Agreement by decision
| decision | agree | total | rate |
|---|---:|---:|---:|
| Oral | 12 | 20 | 60.0% |
| Poster | 251 | 421 | 59.6% |
| Reject | 159 | 279 | 57.0% |
| Spotlight | 51 | 72 | 70.8% |
| Unknown | 7 | 8 | 87.5% |

### Agreement by claim_type
| claim_type | agree | total | rate |
|---|---:|---:|---:|
| methodology | 181 | 284 | 63.7% |
| experimental | 110 | 180 | 61.1% |
| baseline | 51 | 102 | 50.0% |
| presentation | 54 | 86 | 62.8% |
| subjective | 38 | 65 | 58.5% |
| novelty | 27 | 52 | 51.9% |
| quantitative | 15 | 22 | 68.2% |
| other | 4 | 9 | 44.4% |

## B3 (claim vs paper content)

- N = 800
- Exact agreement: **459/800 (57.4%)**

### Label distribution
| label | gpt-5-mini | claude-sonnet-4-6 |
|---|---:|---:|
| Supported | 259 | 206 |
| Partially Supported | 102 | 144 |
| Contradicted | 188 | 127 |
| Not Determined | 251 | 323 |

### Confusion matrix (rows = gpt-5-mini, cols = claude-sonnet-4-6)
| gpt5 \ claude | Supported | Partially Supported | Contradicted | Not Determined | total |
|---|---|---|---|---|---|
| **Supported** | 142 | 43 | 9 | 65 | 259 |
| **Partially Supported** | 19 | 40 | 21 | 22 | 102 |
| **Contradicted** | 18 | 43 | 84 | 43 | 188 |
| **Not Determined** | 27 | 18 | 13 | 193 | 251 |

### Agreement by decision
| decision | agree | total | rate |
|---|---:|---:|---:|
| Oral | 10 | 20 | 50.0% |
| Poster | 240 | 421 | 57.0% |
| Reject | 163 | 279 | 58.4% |
| Spotlight | 41 | 72 | 56.9% |
| Unknown | 5 | 8 | 62.5% |

### Agreement by claim_type
| claim_type | agree | total | rate |
|---|---:|---:|---:|
| methodology | 162 | 284 | 57.0% |
| experimental | 110 | 180 | 61.1% |
| baseline | 58 | 102 | 56.9% |
| presentation | 57 | 86 | 66.3% |
| subjective | 27 | 65 | 41.5% |
| novelty | 23 | 52 | 44.2% |
| quantitative | 16 | 22 | 72.7% |
| other | 6 | 9 | 66.7% |
