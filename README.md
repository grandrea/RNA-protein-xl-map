### script to map RNA-protein crosslinks

Maps output of NuXL RNA-protein crosslinking MS search engine to a PDB file and generates a chimerax pseudobond file.

handles ambiguity of RNA-protein crosslinking mapping by mapping each link to the shortest possible distance.

For example A link from lysine 52 to a dinucleotide GU will be mapped from lysine 52 C alpha to to the closest atom 
in any GU dinucleotide stretch in the model. 