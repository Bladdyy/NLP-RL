1. Think about positional embeddings with semantic tokens, the 1D learned ones might not be the best.
2. Check that transformer networks are properly initialized, maybe there is some improvement to be done here.
3. Run TR 8 BGE exact mean hybrid for the pooling comparison plot (currently only cls exists, token is broken with 7 steps).
4. TR 16 BASE is missing — needed for plots #4 and #6.
5. TR 32 BGE exact cls hybrid is missing — needed for plot #6.
6. MLP 32 BGE exact cls hybrid (`wgkso04c`) is still in-progress (23/100 epochs so far).