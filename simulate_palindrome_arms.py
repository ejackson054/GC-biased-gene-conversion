"""Simulate evolution of ancestral palindrome with ongoing gene conversion between arms

USAGE:  python simulation_arms.py [iteration] [GC_content] [GC_bias] [outdir]

    iteration: Sets random seed for simulation
    GC_content: Starting fraction GC content (e.g. 0.46)
    GC_bias:  Preference for converting to GC base over AT base (e.g. 0.70)
    outdir:  Directory to store results (program will create sub-directory called 'GC_bias_[]_GC_content_[]) 
    
Produces FASTA file containing simulated arms for rhesus macaque, chimpanzee, human, and ancestral palindrome"""


import numpy as np
from scipy.stats import binom
import sys
import os


"""Read in main arguments"""

iteration = int(sys.argv[1])
GC_content = float(sys.argv[2])   
GC_bias = float(sys.argv[3])
outdir = sys.argv[4]


"""Set additional parameters, if desired"""

length = 37000                         #Median human palindrome length, rounded to nearest kb
fraction_differences = 0.00047         #Fraction differences between arms: 0.00047 = median of all 36 conserved X palindromes in human, chimp and rhesus

relative_frequencies = {"AT_to_TA": 0.115, "AT_to_CG": 0.136, "AT_to_GC": 0.566,      #Relative frequencies of different mutation types:  See 
                        "CG_to_AT": 0.227, "CG_to_TA":0.833, "CG_to_GC": 0.210}       #Methods, "Estimation of neutral substitution matrix for simulations"

substitution_rate_branch_34 = 0.0000000120
conversion_rate_branch_34 = (2*substitution_rate_branch_34) / fraction_differences      #Both of these rates are events per nucleotide, per generation
 
substitution_rate_branch_2 = 0.0000000207
conversion_rate_branch_2 = (2*substitution_rate_branch_2) / fraction_differences        #Both of these rates are events per nucleotide, per generation

substitution_rate_branch_1 = 0.0000000186
conversion_rate_branch_1 = (2*substitution_rate_branch_1) / fraction_differences        #Both of these rates are events per nucleotide, per generation

generations_branch_34 = 350000          #Generations in each branch separating human and chimpanzee:  (1 generation / 20 years) * 7 million years = 350000
generations_branch_2 = 1100000          #Generations in branch leading to HC common ancestor:  (1 generation / 20 years) * 22 million years = 1100000
generations_branch_1 =  1450000         #Generations in branch leading to rhesus macaque: (1 generation / 20 years) * 29 million years = 1450000

AT_content = 1 - GC_content
AT_bias = 1 - GC_bias

output_dir = "{0}/GC_bias_{1}_GC_content_{2}".format(outdir, GC_bias, GC_content)

if not os.path.isdir(output_dir):
    os.system("mkdir {0}".format(output_dir))
    
    

"""Define functions"""

def separate_sequence_by_base(sequence):
    
    positions_A = [i for i in range(0,len(sequence)) if sequence[i]=="A"]
    positions_T = [i for i in range(0,len(sequence)) if sequence[i]=="T"]
    positions_C = [i for i in range(0,len(sequence)) if sequence[i]=="C"]
    positions_G = [i for i in range(0,len(sequence)) if sequence[i]=="G"]

    position_dict = {"A": positions_A, "T": positions_T, "C": positions_C, "G": positions_G}
    
    return(position_dict)
    

def calculate_substitution_rates_AT_CG(relative_frequencies, substitution_rate):
    
    #Absolute frequencies
    absolute_frequencies = {}
    
    for i in relative_frequencies.keys():
        absolute_frequencies[i] = relative_frequencies[i]*substitution_rate
    
    #AT and GC rates
    AT_rate = absolute_frequencies["AT_to_TA"] + absolute_frequencies["AT_to_GC"] + absolute_frequencies["AT_to_CG"]
    CG_rate = absolute_frequencies["CG_to_GC"] + absolute_frequencies["CG_to_AT"] + absolute_frequencies["CG_to_TA"]
    
    return(AT_rate, CG_rate)
    
def generate_substitution_matrix(relative_frequencies):
    
    sum_AT = relative_frequencies["AT_to_TA"] + relative_frequencies["AT_to_GC"] + relative_frequencies["AT_to_CG"]
    sum_CG = relative_frequencies["CG_to_GC"] + relative_frequencies["CG_to_AT"] + relative_frequencies["CG_to_TA"]
    
    #Specific rates
    AT_to_TA = relative_frequencies["AT_to_TA"]/sum_AT
    AT_to_GC = relative_frequencies["AT_to_GC"]/sum_AT
    AT_to_CG = relative_frequencies["AT_to_CG"]/sum_AT
    CG_to_GC = relative_frequencies["CG_to_GC"]/sum_CG
    CG_to_AT = relative_frequencies["CG_to_AT"]/sum_CG
    CG_to_TA = relative_frequencies["CG_to_TA"]/sum_CG
    
    #Convert to matrix
    substitution_matrix = {"A": [0, AT_to_TA, AT_to_CG, AT_to_GC], 
                "T": [AT_to_TA, 0, AT_to_CG, AT_to_GC], 
                "G": [CG_to_AT, CG_to_TA, 0, CG_to_GC], 
                "C": [CG_to_AT, CG_to_TA, CG_to_GC, 0]}
    
    return(substitution_matrix)


    
def assign_conversion_prob(base_1, base_2):
    
    c1 = (base_1 in ["G","C"]) and (base_2 in ["G","C"])
    c2 = (base_1 in ["A","T"]) and (base_2 in ["A","T"])
    c3 = (base_1 in ["G","C"]) and  (base_2 in ["A","T"])
    c4  = (base_1 in ["A","T"]) and (base_2 in ["G","C"])
    
    if c1 or c2:
        
        prob = [0.5, 0.5]
        
    elif c3:
        
        prob = [GC_bias, AT_bias]
    
    elif c4:
        
        prob = [AT_bias, GC_bias]
    
    return(prob)
        
    

def mutate_arm(arm, possible_positions, number_mut, position_dict):
    
    global substitution_matrix
    
    pos = np.random.choice(possible_positions, size=number_mut)
    
    for i in pos:
        old = arm[i]
        new = np.random.choice(bases,p = substitution_matrix[old])
        arm[i] = new
        position_dict[old].remove(i)
        position_dict[new].append(i)
        
  
def convert_arm(arm_1, arm_2, number_conv, position_dict_1, position_dict_2):  
    
    possible_pos = list(range(0,len(arm_1)))
    
    pos = np.random.choice(possible_pos, size=number_conv)
        
    for i in pos:
        
        prob = assign_conversion_prob(arm_1[i], arm_2[i])
        
        converted_base = np.random.choice([arm_1[i], arm_2[i]],p = prob)
        
        #Update list of positions that contains each base!
        if converted_base == arm_1[i]:
            position_dict_2[arm_2[i]].remove(i)
            position_dict_2[arm_1[i]].append(i)
            arm_2[i] = arm_1[i]
        
        else: 
            position_dict_1[arm_1[i]].remove(i)
            position_dict_1[arm_2[i]].append(i)
            arm_1[i] = arm_2[i]

         
   
     
def simulate_one_generation(arm_1, arm_2, conversion_rate, substitution_rate_AT, substitution_rate_CG, position_dict_arm_1, position_dict_arm_2):
    
    #Select positions to mutate
    #You have n = 37000, p = mutation rate; draw once from binomial distribution to see how many mutations you get (will usually be 0)
    number_mut_A_1 = binom.rvs(len(position_dict_arm_1["A"]), substitution_rate_AT, size = 1)
    number_mut_T_1 = binom.rvs(len(position_dict_arm_1["T"]), substitution_rate_AT, size = 1)
    number_mut_C_1 = binom.rvs(len(position_dict_arm_1["C"]), substitution_rate_CG, size = 1)
    number_mut_G_1 = binom.rvs(len(position_dict_arm_1["G"]), substitution_rate_CG, size = 1)
    
    number_mut_A_2 = binom.rvs(len(position_dict_arm_2["A"]), substitution_rate_AT, size = 1)
    number_mut_T_2 = binom.rvs(len(position_dict_arm_2["T"]), substitution_rate_AT, size = 1)
    number_mut_C_2 = binom.rvs(len(position_dict_arm_2["C"]), substitution_rate_CG, size = 1)
    number_mut_G_2 = binom.rvs(len(position_dict_arm_2["G"]), substitution_rate_CG, size = 1)
    
    
    dictionary_1 = {"A": [position_dict_arm_1["A"], number_mut_A_1], "T": [position_dict_arm_1["T"], number_mut_T_1],
    "C": [position_dict_arm_1["C"], number_mut_C_1], "G": [position_dict_arm_1["G"], number_mut_G_1]}
    
    dictionary_2 = {"A": [position_dict_arm_2["A"], number_mut_A_2], "T": [position_dict_arm_2["T"], number_mut_T_2],
    "C": [position_dict_arm_2["C"], number_mut_C_2], "G": [position_dict_arm_2["G"], number_mut_G_2]}
    
    for base in dictionary_1.keys():
        
        possible_positions = dictionary_1[base][0]
        number_mut = dictionary_1[base][1]
        
        if dictionary_1[base][1]!=0:
        
            mutate_arm(arm_1, possible_positions, number_mut, position_dict_arm_1)
    
    for base in dictionary_2.keys():
        
        possible_positions = dictionary_2[base][0]
        number_mut = dictionary_2[base][1]
        
        if dictionary_2[base][1]!=0:
        
            mutate_arm(arm_2, possible_positions, number_mut, position_dict_arm_2)
    

    #Select positions for gene conversion
    number_conv = binom.rvs(len(arm_1), conversion_rate, size = 1)
    
    if number_conv != 0:
        
        convert_arm(arm_1, arm_2, number_conv, position_dict_arm_1, position_dict_arm_2)
        
        
"""Main script"""

#Get mutation spectrum
substitution_matrix = generate_substitution_matrix(relative_frequencies)
substitution_rate_branch34_AT, substitution_rate_branch34_CG = calculate_substitution_rates_AT_CG(relative_frequencies, substitution_rate_branch_34)
substitution_rate_branch1_AT, substitution_rate_branch1_CG = calculate_substitution_rates_AT_CG(relative_frequencies, substitution_rate_branch_1)
substitution_rate_branch2_AT, substitution_rate_branch2_CG = calculate_substitution_rates_AT_CG(relative_frequencies, substitution_rate_branch_2)

#Initialize arms
bases = ["A","T","G","C"]
p = [AT_content/2, AT_content/2, GC_content/2, GC_content/2]

np.random.seed(iteration) 
initial_arm_1 = np.random.choice(bases, size=length, p=p)

#Make Arm 2 different 
initial_arm_2 = []

p_same = 1-fraction_differences
p_diff = fraction_differences/3

difference_probs = {"A": [p_same, p_diff, p_diff, p_diff], 
"T": [p_diff, p_same, p_diff, p_diff], 
"G": [p_diff, p_diff, p_same, p_diff], 
"C": [p_diff, p_diff, p_diff, p_same]}

for i in initial_arm_1:
    
    choice = np.random.choice(bases, p=difference_probs[i])
    initial_arm_2.append(choice)
      
#Make a separate identical variables for human-chimp arms, and rhesus arms
arm_1_human_chimp = [i for i in initial_arm_1]     
arm_2_human_chimp = [i for i in initial_arm_2]  
arm_1_rhesus = [i for i in initial_arm_1]
arm_2_rhesus = [i for i in initial_arm_2]
arm_1_ancestral = [i for i in initial_arm_1]
arm_2_ancestral = [i for i in initial_arm_2]  

#Make lists of positions for each base
position_dict_rhesus_arm_1 = separate_sequence_by_base(initial_arm_1)
position_dict_rhesus_arm_2 = separate_sequence_by_base(initial_arm_2)

position_dict_human_chimp_arm_1 = separate_sequence_by_base(initial_arm_1)
position_dict_human_chimp_arm_2 = separate_sequence_by_base(initial_arm_2)


#Branch 1:  Rhesus macaque simulations
for i in list(range(0,generations_branch_1)):
    
    simulate_one_generation(arm_1_rhesus, arm_2_rhesus, conversion_rate_branch_1, substitution_rate_branch1_AT, 
    substitution_rate_branch1_CG, position_dict_rhesus_arm_1, position_dict_rhesus_arm_2)

#Branch 2:  Human-chimpanzee common ancestor simulations
for i in list(range(0,generations_branch_2)):
    
    simulate_one_generation(arm_1_human_chimp, arm_2_human_chimp, conversion_rate_branch_2, substitution_rate_branch2_AT, 
    substitution_rate_branch2_CG, position_dict_human_chimp_arm_1, position_dict_human_chimp_arm_2)
    

#Update arms:  Chimp and human are now diverged!
arm_1_human = [i for i in arm_1_human_chimp]  
arm_2_human = [i for i in arm_2_human_chimp]  
arm_1_chimp = [i for i in arm_1_human_chimp]  
arm_2_chimp = [i for i in arm_2_human_chimp]  

#Update position dictionaries
position_dict_human_arm_1 = separate_sequence_by_base(arm_1_human)
position_dict_human_arm_2 = separate_sequence_by_base(arm_2_human)

position_dict_chimp_arm_1 = separate_sequence_by_base(arm_1_chimp)
position_dict_chimp_arm_2 = separate_sequence_by_base(arm_2_chimp)


#Branches 3 and 4:  Human and chimpanzee simulations
for i in list(range(0,generations_branch_34)):
    
    simulate_one_generation(arm_1_human, arm_2_human, conversion_rate_branch_34, substitution_rate_branch34_AT,
    substitution_rate_branch34_CG, position_dict_human_arm_1, position_dict_human_arm_2)

for i in list(range(0,generations_branch_34)):
    
    simulate_one_generation(arm_1_chimp, arm_2_chimp, conversion_rate_branch_34, substitution_rate_branch34_AT,
    substitution_rate_branch34_CG, position_dict_chimp_arm_1, position_dict_chimp_arm_2)
      
#Write results to output file
arm_1_human = "".join(arm_1_human)
arm_2_human = "".join(arm_2_human)

arm_1_chimp = "".join(arm_1_chimp)
arm_2_chimp = "".join(arm_2_chimp)

arm_1_rhesus = "".join(arm_1_rhesus)
arm_2_rhesus = "".join(arm_2_rhesus)

arm_1_ancestral = "".join(arm_1_ancestral)
arm_2_ancestral = "".join(arm_2_ancestral)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

with open("{0}/Iteration_{1}.fasta".format(output_dir, iteration),"w") as new:
    new.write(""">Arm_1_rhesus\n{0}\n>Arm_2_rhesus\n{1}\n>Arm_1_human\n{2}\n>Arm_2_human\n{3}\n>Arm_1_chimp\n
    {4}\n>Arm_2_chimp\n{5}\n>Arm_1_ancestral\n{6}\n>Arm_2_ancestral\n{7}""".format(arm_1_rhesus,arm_2_rhesus, 
    arm_1_human, arm_2_human, arm_1_chimp, arm_2_chimp, arm_1_ancestral, arm_2_ancestral)) 