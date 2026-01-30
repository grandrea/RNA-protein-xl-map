from Bio.PDB import *
from pyteomics.openms import idxml

parser = PDBParser()

structure_file = "260126_nsun2_tRNA_fitted_on640.pdb"
nuxl_idxml_file = "E260114_05_HT_AD_120_Leroy_NSUN2_tRNA_XLMS_Peak2_new.idXML"


RNA_chain  = "B"
protein_chain = "A"

structure = parser.get_structure("structure", structure_file)



structure[0]["B"][13]["P"] - structure[0]["A"][45]["CA"]

carbon_atoms = [
    atom for atom in a.get_atoms()
    if atom.get_name().startswith("C") and "'" not in atom.get_name()
]