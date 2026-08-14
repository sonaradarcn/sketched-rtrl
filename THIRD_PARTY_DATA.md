# Third-party data redistributed with this repository

Everything under `skrtrl/data/` is third-party measured data. It is **not** covered by the MIT
licence in `LICENSE`, which applies only to the code and to the result records under `results/`.
Two files are redistributed here so that the two real benchmarks in the paper run out of the box;
the loader in `skrtrl/tasks.py` raises if a file is missing and never substitutes a synthetic
surrogate.

If you use either series, cite the original source, not this repository.

---

## `skrtrl/data/sunspot.txt`

| | |
|---|---|
| Series | Monthly mean total sunspot number |
| Product | `SN_m_tot_V2.0` (version 2.0). **Not** the 13-month smoothed `SN_ms_tot_V2.0` |
| Source | WDC-SILSO, Royal Observatory of Belgium, Brussels — <https://www.sidc.be/SILSO/> |
| Obtained from | <https://www.sidc.be/SILSO/INFO/snmtotcsv.php> (series column) |
| Retrieved | 2026-06 |
| Contents here | 3329 raw monthly means, one per line, 1749-01 through 2026-05 |
| Bytes / SHA-256 | 17378 / `73cc4b18cee5d811bf94db3882f831b4dad20f4fd2206b6b008305f469abae6e` |
| Terms | SILSO data are distributed for free use with the attribution requirement stated by SILSO; the standard acknowledgement is *"Source: WDC-SILSO, Royal Observatory of Belgium, Brussels"*. Consult <https://www.sidc.be/SILSO/> for the current terms before redistributing further. |

Identity check, so the product cannot be confused with the smoothed one: the first twelve values
are 96.7 / 104.3 / 116.7 / 92.8 / 141.7 / 139.2 / 158.0 / 110.5 / 126.5 / 125.8 / 264.3 / 142.0,
which is `SN_m_tot` for 1749-01..1749-12. `SN_ms_tot` is undefined (`-1.0`) for the first six
months and reads 135.9 at 1749-07. The mean absolute month-to-month change here is 19.2 (max
156.5), far larger than any 13-month moving average could produce.

## `skrtrl/data/laser.txt`

| | |
|---|---|
| Series | Far-infrared laser intensity, chaotic regime |
| Source | Data set A of the Santa Fe time-series prediction competition (Weigend & Gershenfeld, eds., *Time Series Prediction*, Addison-Wesley, 1994); measurements by U. Hübner et al. |
| Retrieved | 2026-06, from a mirror of the competition archive; the original competition server is no longer maintained |
| Contents here | The 1000-point training segment, one integer per line |
| Bytes / SHA-256 | 3138 / `01cdac6311bfb4d283c2ff9634a44c96ae1fa5c852fc62415fafbf5ea012327a` |
| Terms | **No explicit redistribution licence is known to us.** The competition data have circulated for three decades as a standard public benchmark and are mirrored in many packages, but we have not located a licence text attached to them. The copy here is provided so that the results in the accompanying paper can be reproduced, and it is not offered under any licence we can name. Treat it as third-party material with undetermined redistribution terms rather than as licensed data. |

**If you are the rights holder** for either series and object to redistribution, open an issue and
the file will be removed; `skrtrl/tasks.py` already fails loudly on a missing file, so the code
keeps working from a locally supplied copy dropped at the same path.

**On the checksums.** Both figures above are of the exact bytes git stores and checks out, which
is LF-terminated: `.gitattributes` marks `skrtrl/data/*.txt` as `-text` so no platform rewrites
the line endings and the hashes hold on Windows, macOS and Linux alike. If you obtained a copy
elsewhere and the hash differs, normalise its line endings to LF before concluding the contents
differ.
