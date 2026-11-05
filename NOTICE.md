# Attribution and release status

This source snapshot accompanies *Busemannformer: Horospherical Self-Attention
for Hyperbolic Graph Transformers*. The public source repository is
[LionelYao/Busemannformer](https://github.com/LionelYao/Busemannformer).
No archival DOI is asserted here.

The original model docstring is retained. It acknowledges conventions from
HGCN (Hyperboloid APIs), Hypformer (hyperbolic layers and optional HTC
projections), and Graphormer (optional graph bias and graph token).
No third-party source tree, pretrained weights, or benchmark data is bundled.
Dataset users must obtain the data from its original provider and observe its
terms; the NPZ examples avoid redistributing those assets.

`SOURCE_PROVENANCE.json` records the hashes of the original copied sources.
The three model/split modules are unchanged. The copied LP test only changes
the skip condition of its historical H200-only test to avoid failing on other
GPU models. Portable training scripts, examples, packaging, and their tests are
new release utilities, not the historical cluster launchers.

The intended policy is to retain the original project license. No project-level
license was found in the supplied source, the older local copies, or the checked
original Git history. The licenses of bundled third-party projects are not used
as substitutes. The original license, contributor rights, and upstream
obligations still need confirmation. Public availability does not resolve
these questions; no new license grant is implied by this notice.
