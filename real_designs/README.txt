Real ProteinMPNN output, generated locally with the weights shipped in
github.com/dauparas/ProteinMPNN (commit 8907e66).

  vanilla/   v_48_020 default weights,  T = 0.1 / 0.2 / 0.3, 30 per temperature
  soluble/   soluble-trained weights,   T = 0.1 / 0.2 / 0.3, 30 per temperature
  hot/       v_48_020 default weights,  T = 0.5 / 0.8,       20 per temperature

Six real kinase catalytic-domain backbones, chain A, seed 37 (41 for hot):
tyk2, cmet, p38, syk, cdk2, cdk8. PDBs from the OpenFF protein-ligand
benchmark (complete N/CA/C/O backbones); cdk2 and cdk8 carry a cyclin chain,
held fixed as context.

The FIRST record of every file is the NATIVE sequence, not a design. It has
designed_chains= in its header and no T=. Read these with

    gens = md.load_generator_output(["vanilla", "soluble"])   # skips natives
    md.load_generator_output(["vanilla"], include_native=True)  # keeps them

rather than parsing them as plain FASTA, or the native lands in your design
set as a guaranteed miss.
