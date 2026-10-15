"""Final captions of the generated table fragments, as printed in the manuscript.

make_r2_d4_tables.py and make_r2_w1b_tables.py compute the table bodies; the caption line of
each fragment is replaced by the one below, so a regenerated fragment matches the manuscript
line for line apart from the comment header.
"""
import re

CAPTIONS = {
    'tab:r2fidelity':
        "\\caption{\\textbf{Gradient cosine to the exact RTRL gradient on the four diagnostic tasks} ($n{=}64$; 10 seeds ($0$--$9$) $\\times$ 20\\,000 steps at the per-pair tuned rates of Table~\\mainnum{tab:r2lr}{13} of the main paper; protocol and table conventions in Section~\\mainnum{sec:exp-setup}{6.1}). Paired Wilcoxon tests with Holm correction, paired bootstrap $95\\%$ confidence intervals and Cohen's $d_z$ for every one of these comparisons are in \\Cref{tab:r2stats,tab:r2stats2,tab:r2stats3}.}",
    'tab:r2timeseries':
        '\\caption{\\textbf{Online next-step prediction on the three chaotic systems} ($n{=}64$; 10 seeds ($0$--$9$) $\\times$ 20\\,000 steps at the per-pair tuned rates of Table~\\mainnum{tab:r2lr}{13} of the main paper; protocol and table conventions in Section~\\mainnum{sec:exp-setup}{6.1}). The per-system columns are the task metric (NMSE, lower is better) and the last the gradient cosine to exact RTRL, pooled over every seed and every system of the block at once (30 runs per entry, higher is better); per-system cosines and the paired statistics of every comparison are in \\Cref{tab:r2stats,tab:r2stats2,tab:r2stats3}.}',
    'tab:r2realts':
        '\\caption{\\textbf{Online next-step prediction on the two real series} (monthly Sunspot and Santa~Fe laser, one-off non-causal fixed-window normalisation, $n{=}64$; 10 seeds ($0$--$9$) $\\times$ 20\\,000 steps at the per-pair tuned rates of Table~\\mainnum{tab:r2lr}{13} of the main paper; protocol and table conventions in Section~\\mainnum{sec:exp-setup}{6.1}). The per-system columns are the task metric (NMSE, lower is better) and the last the gradient cosine to exact RTRL, pooled over every seed and every system of the block at once (20 runs per entry, higher is better); per-system cosines and the paired statistics of every comparison are in \\Cref{tab:r2stats,tab:r2stats2,tab:r2stats3}.}',
    'tab:r2lr':
        "\\caption{\\textbf{Learning rate selected for every (task, estimator) pair by the two-stage procedure}, and therefore used for every entry of Tables~S6--S8 of the supplementary material, with TBPTT's tuned truncation window $k$ printed beside the rate. The selection score and the search grid are given in \\Cref{sec:exp-setup}.}",
    'tab:r2stats':
        "\\caption{\\textbf{Paired comparisons of the full matrix, \\skrtrl{} $r{=}4$} (part 1 of 3, the 90 comparisons at this rank): the paired mean advantage $\\Delta$ with a $95\\%$ bootstrap interval, the Holm-corrected Wilcoxon $p_{\\mathrm{H}}$, the effect size $d_z$ and Cliff's $\\delta$, over $10$ paired seeds. Column definitions, conventions and the sign convention on $\\Delta$ are in \\Cref{app:suppstats}.}",
    'tab:r2stats2':
        "\\caption{\\textbf{Paired comparisons of the full matrix, \\skrtrl{} $r{=}16$} (part 2 of 3, the 90 comparisons at this rank): the paired mean advantage $\\Delta$ with a $95\\%$ bootstrap interval, the Holm-corrected Wilcoxon $p_{\\mathrm{H}}$, the effect size $d_z$ and Cliff's $\\delta$, over $10$ paired seeds. Column definitions, conventions and the sign convention on $\\Delta$ are in \\Cref{app:suppstats}.}",
    'tab:r2stats3':
        "\\caption{\\textbf{Paired comparisons of the full matrix, \\skrtrl{} $r{=}32$} (part 3 of 3, the 90 comparisons at this rank): the paired mean advantage $\\Delta$ with a $95\\%$ bootstrap interval, the Holm-corrected Wilcoxon $p_{\\mathrm{H}}$, the effect size $d_z$ and Cliff's $\\delta$, over $10$ paired seeds. Column definitions, conventions and the sign convention on $\\Delta$ are in \\Cref{app:suppstats}.}",
    'tab:tightness':
        "\\caption{\\textbf{Certificate tightness in certified-contractive phases}, over the full task set (the clip sweep of Table~\\mainnum{tab:r2cert}{5} of the main paper: \\skrtrl{} $r{=}16$, nine tasks, clip $\\in\\{0.35,0.7,0.9\\}$, five seeds, $20{,}000$ steps, Table~\\mainnum{tab:protocol}{12}). Each row pools the logged steps whose certified spectral bound $\\rhobar_t$ lies in the stated interval and reports the \\emph{median} measured ratio $e_t/\\fnorm{E_t}$ against the geometric factor $1/(1-\\rhobar)$ evaluated at the row's pooled mean $\\rhobar$; the reading of the top bin and the absent copy row are discussed in \\Cref{app:supp-cert-tight}.}",
    'tab:adaptive':
        '\\caption{\\textbf{Certificate-safe adaptive rank against the same-trajectory fixed-rank sweep} ($n{=}64$, clip $0.5$, learning rate $10^{-3}$, $12{,}000$ steps, seeds $0$--$2$, mean $\\pm$ sample s.d.). The fixed rows sweep $r$ along one trajectory and the adaptive rows are the controller of \\Cref{sec:exp-adaptive} at $[4,32]$ and at $[4,64]$, with zero Theorem-1 violations over the 648{,}000 certificate checks behind the whole sweep; the columns are defined in \\Cref{sec:exp-adaptive}.}',
}


def apply(text):
    """Replace each caption line that is directly followed by a known label line."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        m = re.match(r"\\label\{([^}]+)\}", lines[i + 1])
        if lines[i].startswith("\\caption{") and m and m.group(1) in CAPTIONS:
            lines[i] = CAPTIONS[m.group(1)]
    return "\n".join(lines)
