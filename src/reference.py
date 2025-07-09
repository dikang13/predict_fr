# Set on a list of neurons that have good signal and number of traces
fig4_neuron_classes = [
    "AVB", "RIB", "RIC", "RID", "AUA", "AVJ", "AVK", "AIM", "AIY", "AIA",
    "AVA", "AVE", "AIB", "RIM", "AVL", "RIF", "RIV", "ADA", "AVD", "RMF",
    "RIA", "AVH", "RIR", "RIS", "RIH", "AIN", "RIP", "AIZ", "URB", "ALA",
    "RMG", "RMD", "RMDD", "RMDV", "RME", "RMEV", "RMED", "SAADL", "SAADR",
    "SAAV", "SMBV", "SMBD", "SMDV", "SMDD", "SIAV", "SIAD", "SIBV", "SIBD",
    "VB02", "ASJ", "IL1L", "IL1R", "IL1D", "IL1V", "URYD", "URYV", "BAG",
    "ASG", "CEPD", "CEPV", "OLL", "OLQD", "OLQV", "IL2L", "IL2R", "IL2D",
    "IL2V", "URAD", "URAV", "ADE", "FLP", "AQR", "URX", "ADL", "ASH", "ASEL",
    "ASER", "ASI", "AFD", "ASK", "AWA", "AWB", "AWC", "I1", "I2", "I3", "I4",
    "I5", "I6", "NSM", "M1", "M3", "M4", "M5", "MC", "MI"
]

# # Define neuron classes by block
# neuron_blocks = {
#     # Block 1 - Command interneurons (BRIGHT RED - high prominence)
#     1: ["AVB", "RIB", "RIC", "RID", "AUA", "AVJ", "AVK", "AIM", "AIY", "AIA"],
    
#     # Block 2 - Forward/backward control (ELECTRIC BLUE - high prominence)  
#     2: ["AVA", "AVE", "AIB", "RIM"],
    
#     # Block 3 - Head neurons and motor control (MUTED PURPLE)
#     3: ["AVL", "RIF", "RIV", "ADA", "AVD", "RMF", "RIA", "AVH", "RIR", "RIS", 
#         "RIH", "AIN", "RIP", "AIZ", "URB", "ALA", "RMG", "RMD", "RMDD", 
#         "RMDV", "RME", "RMEV", "RMED"],
    
#     # Block 4 - Sublateral neurons (MUTED TEAL)
#     4: ["SAADL", "SAADR", "SAAV", "SMBV", "SMBD", "SMDV", "SMDD", "SIAV", 
#         "SIAD", "SIBV", "SIBD", "VB02"],
    
#     # Block 5 - Amphid and phasmid neurons (MUTED ORANGE)
#     5: ["ASJ", "IL1L", "IL1R", "IL1D", "IL1V", "URYD", "URYV", "BAG", "ASG",
#         "CEPD", "CEPV", "OLL", "OLQD", "OLQV", "IL2L", "IL2R", "IL2D", 
#         "IL2V", "URAD", "URAV", "ADE", "FLP", "AQR", "URX", "ADL"],
    
#     # Block 6 - Chemosensory neurons (MUTED GREEN)
#     6: ["ASH", "ASEL", "ASER", "ASI", "AFD", "ASK", "AWA", "AWB", "AWC"],
    
#     # Block 7 - Pharyngeal neurons (BRIGHT GOLD - high prominence)
#     7: ["I1", "I2", "I3", "I4", "I5", "I6", "NSM", "M1", "M3", "M4", "M5", "MC", "MI"]
# }

# # Define color scheme - prominent blocks get vivid colors, others get muted
# block_colors = {
#     1: 'red',  # Bright Red - Command interneurons (PROMINENT)
#     2: 'blue',  # Electric Blue - Forward/backward control (PROMINENT)
#     3: '#9970AB',  # Muted Purple - Head neurons and motor control
#     4: 'pink',  # Muted Teal - Sublateral neurons  
#     5: 'yellow',  # Muted Orange - Amphid and phasmid neurons
#     6: 'cyan',  # Muted Green - Chemosensory neurons
#     7: 'lime',  # Bright Gold - Pharyngeal neurons (PROMINENT)
# }

# # Create the color dictionary for each neuron
# colors_neuron_classes = {}

# for block_num, neurons in neuron_blocks.items():
#     color = block_colors[block_num]
#     for neuron in neurons:
#         colors_neuron_classes[neuron] = color

# # Print the color assignments for verification
# print("Neuron Color Assignments:")
# print("=" * 50)

# for block_num, neurons in neuron_blocks.items():
#     prominence = " (PROMINENT)" if block_num in [1, 2, 7] else ""
#     print(f"\nBlock {block_num}{prominence} - Color: {block_colors[block_num]}")
#     print(f"Neurons: {', '.join(neurons)}")

# # Create a visualization of the color scheme
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# # Plot color swatches for each block
# y_positions = np.arange(len(neuron_blocks))
# bar_height = 0.8

# for i, (block_num, neurons) in enumerate(neuron_blocks.items()):
#     color = block_colors[block_num]
#     prominence = " (PROMINENT)" if block_num in [1, 2, 7] else ""
    
#     # Create bar
#     bar = ax.barh(y_positions[i], 1, height=bar_height, 
#                   color=color, alpha=0.8, edgecolor='black', linewidth=1)
    
#     # Add block label
#     ax.text(0.5, y_positions[i], f'Block {block_num}{prominence}\n({len(neurons)} neurons)', 
#             ha='center', va='center', fontweight='bold', fontsize=10)

# # Customize the plot
# ax.set_xlim(0, 1)
# ax.set_ylim(-0.5, len(neuron_blocks) - 0.5)
# ax.set_yticks([])
# ax.set_xticks([])
# ax.set_title('Neuron Class Color Scheme\n(Blocks 1, 2, and 7 are Prominent)', 
#              fontsize=14, fontweight='bold', pad=20)

# # Remove spines
# for spine in ax.spines.values():
#     spine.set_visible(False)

# plt.tight_layout()
# plt.show()

# # Alternative: If you want hex codes in a simple dictionary format
# print("\n" + "=" * 50)
# print("Dictionary format for direct use:")
# print("colors_neuron_classes = {")
# for block_num, neurons in neuron_blocks.items():
#     color = block_colors[block_num]
#     print(f"    # Block {block_num}")
#     for neuron in neurons:
#         print(f"    '{neuron}': '{color}',")
#     print()
# print("}")

colors_neuron_classes = {
    # Block 1
    'AVB': 'red',
    'RIB': 'red',
    'RIC': 'red',
    'RID': 'red',
    'AUA': 'red',
    'AVJ': 'red',
    'AVK': 'red',
    'AIM': 'red',
    'AIY': 'red',
    'AIA': 'red',

    # Block 2
    'AVA': 'blue',
    'AVE': 'blue',
    'AIB': 'blue',
    'RIM': 'blue',

    # Block 3
    'AVL': '#9970AB',
    'RIF': '#9970AB',
    'RIV': '#9970AB',
    'ADA': '#9970AB',
    'AVD': '#9970AB',
    'RMF': '#9970AB',
    'RIA': '#9970AB',
    'AVH': '#9970AB',
    'RIR': '#9970AB',
    'RIS': '#9970AB',
    'RIH': '#9970AB',
    'AIN': '#9970AB',
    'RIP': '#9970AB',
    'AIZ': '#9970AB',
    'URB': '#9970AB',
    'ALA': '#9970AB',
    'RMG': '#9970AB',
    'RMD': '#9970AB',
    'RMDD': '#9970AB',
    'RMDV': '#9970AB',
    'RME': '#9970AB',
    'RMEV': '#9970AB',
    'RMED': '#9970AB',

    # Block 4
    'SAADL': 'pink',
    'SAADR': 'pink',
    'SAAV': 'pink',
    'SMBV': 'pink',
    'SMBD': 'pink',
    'SMDV': 'pink',
    'SMDD': 'pink',
    'SIAV': 'pink',
    'SIAD': 'pink',
    'SIBV': 'pink',
    'SIBD': 'pink',
    'VB02': 'pink',

    # Block 5
    'ASJ': 'yellow',
    'IL1L': 'yellow',
    'IL1R': 'yellow',
    'IL1D': 'yellow',
    'IL1V': 'yellow',
    'URYD': 'yellow',
    'URYV': 'yellow',
    'BAG': 'yellow',
    'ASG': 'yellow',
    'CEPD': 'yellow',
    'CEPV': 'yellow',
    'OLL': 'yellow',
    'OLQD': 'yellow',
    'OLQV': 'yellow',
    'IL2L': 'yellow',
    'IL2R': 'yellow',
    'IL2D': 'yellow',
    'IL2V': 'yellow',
    'URAD': 'yellow',
    'URAV': 'yellow',
    'ADE': 'yellow',
    'FLP': 'yellow',
    'AQR': 'yellow',
    'URX': 'yellow',
    'ADL': 'yellow',

    # Block 6
    'ASH': 'cyan',
    'ASEL': 'cyan',
    'ASER': 'cyan',
    'ASI': 'cyan',
    'AFD': 'cyan',
    'ASK': 'cyan',
    'AWA': 'cyan',
    'AWB': 'cyan',
    'AWC': 'cyan',

    # Block 7
    'I1': 'lime',
    'I2': 'lime',
    'I3': 'lime',
    'I4': 'lime',
    'I5': 'lime',
    'I6': 'lime',
    'NSM': 'lime',
    'M1': 'lime',
    'M3': 'lime',
    'M4': 'lime',
    'M5': 'lime',
    'MC': 'lime',
    'MI': 'lime',

}