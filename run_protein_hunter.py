import os
import subprocess

boltz_ph_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "boltz_ph", "design.py"))
chai_ph_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "chai_ph", "design.py"))


# Protein Hunter (Boltz Edition ⚡) 
# protein binding design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "2",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "0",
    "--name", "PDL1_mix_aa",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)


# re-designing existing binder design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "2", 
    "--num_cycles", "7",
    "--seq", "GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "0",
    "--name", "PDL1_mix_aa_refiner",
    "--high_iptm_threshold", "0.8",
    "--use_msa_for_af3",
    "--plot",
]
subprocess.run(cmd, check=True)



cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "single",
    "--gpu_id", "0",
    "--name", "PDL1_mix_aa_template",
    "--template_path", "8ZNL",
    "--template_cif_chain_id", "B",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)

# contact condition design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "2",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "1",
    "--name", "PDL1_contact_condition",
    "--contact_residues", "1,2,5,10",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)



#multimer binding design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK:FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "0",
    "--name", "PDL1_double_mix_aa",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)

#protein binder with contact residues
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "0",
    "--contact_residues", "29,277,279,293,294,295,318,319,320,371",
    "--name", "PDL1_contact_residues_mix_aa",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)

# #protein + small molecule binding design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--msa_mode", "mmseqs",
    "--gpu_id", "0",
    "--name", "PDL1_SAM_mix_aa",
    "--ligand_ccd", "SAM",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)


#nucleic acid binding design
cmd = [
    "python", boltz_ph_script_path,
    "--num_designs", "3",
    "--num_cycles", "7",
    "--gpu_id", "0",
    "--name", "rna_mix_aa",
    "--nucleic_seq", "AGAGAGA",
    "--nucleic_type", "rna",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)




#Protein Hunter (Chai Edition ☕)

#unconditional design
cmd = [
    "python", chai_ph_script_path,
    "--jobname", "unconditional_design",
    "--percent_X", "0",
    "--min_protein_length", "150",
    "--max_protein_length", "150",
    "--n_trials", "3",
    "--n_cycles", "7",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "templates",
    "--repredict",
    "--omit_aa", "",
    "--temperature", "0.1",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)

#unconditional design
cmd = [
    "python", chai_ph_script_path,
    "--jobname", "unconditional_design_negative_alanine_bias",
    "--percent_X", "0",
    "--min_protein_length", "150",
    "--max_protein_length", "150",
    "--n_trials", "1",
    "--n_cycles", "5",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "templates",
    "--repredict",
    "--omit_aa", "",
    "--temperature", "0.1",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
    "--alanine_bias",
]
subprocess.run(cmd, check=True)

#protein binder design
cmd = [
    "python", chai_ph_script_path,
    "--jobname", "PDL1_binder",
    "--percent_X", "50",
    "--min_protein_length", "90",
    "--max_protein_length", "150",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--n_trials", "3",
    "--n_cycles", "7",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "templates",
    "--repredict",
    "--omit_aa", "",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
    "--use_msa_for_af3",
]
subprocess.run(cmd, check=True)


# re-designing existing binder design
cmd = [
    "python", chai_ph_script_path,
    "--jobname", "PDL1_binder_refiner",
    "--percent_X", "50",
    "--seq", "GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--n_trials", "3",
    "--n_cycles", "7",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "templates",
    "--repredict",
    "--omit_aa", "",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
    "--use_msa_for_af3",
]
subprocess.run(cmd, check=True)


#protein cyclic binder design

cmd = [
    "python", chai_ph_script_path,
    "--jobname", "PDL1_cyclic_binder",
    "--percent_X", "50",
    "--seq", "",
    "--min_protein_length", "10",
    "--max_protein_length", "20",
    "--target_seq", "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK",
    "--n_trials", "3",
    "--n_cycles", "7",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "templates",
    "--repredict",
    "--omit_aa", "",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
    "--use_msa_for_af3",
]
subprocess.run(cmd, check=True)


#ligand binder design
cmd = [
    "python", chai_ph_script_path,
    "--jobname", "ligand_binder",
    "--percent_X", "50",
    "--seq", "",
    "--min_protein_length", "130",
    "--max_protein_length", "150",
    "--target_seq", "O=C(NCc1cocn1)c1cnn(C)c1C(=O)Nc1ccn2cc(nc2n1)c1ccccc1",
    "--n_trials", "3",
    "--n_cycles", "7",
    "--n_recycles", "3",
    "--n_diff_steps", "200",
    "--hysteresis_mode", "esm",
    "--repredict",
    "--omit_aa", "",
    "--temperature", "0.01",
    "--high_iptm_threshold", "0.7",
    "--scale_temp_by_plddt",
    "--render_freq", "100",
    "--gpu_id", "0",
]
subprocess.run(cmd, check=True)