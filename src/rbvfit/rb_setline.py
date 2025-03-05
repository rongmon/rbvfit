from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
from astropy.io import ascii
from pkg_resources import resource_filename


'''
 Function to read in atomic line information for a given rest frame  wavelength.
                           Or 
 For the line matching the closest wavelength. 

 Input :
 		   lambda_rest :-  Rest Frame wavelength (in \AA) of the line to match
   		    method     :-    'closest' ->  If set will match the closest line.
   					    	  'Exact'  ->  If set will match the exact wavelength.
 
 Output:    dic :- Dictionary with fval,lambda and species name.

 Example:   str=rb_setline(2796.3,'closest')


Written By: Rongmon Bordoloi 				Jan 2018, Python 2.7
Edit:       Rongmon Bordoloi                            Sep 2018, Depreciated kwargs to be compatible with python 3
'''
def rb_setline(lambda_rest, method, linelist='atom'):
    """
    Retrieves atomic line information for a given rest-frame wavelength.

    Parameters:
    lambda_rest (float): Rest-frame wavelength (in Å) of the line to match.
    method (str): 'closest' to match the closest line, 'Exact' to match the exact wavelength.
    linelist (str, optional): Specifies the line list ('atom', 'LLS', 'LLS Small', 'DLA'). Default is 'atom'.

    Returns:
    dict: A dictionary containing matched line information (wavelength, f-value, name, and gamma if applicable).
    """
    line_str = read_line_list(linelist)
    num_lines = len(line_str)
    
    # Pre-allocate arrays for efficiency
    wavelist = np.zeros(num_lines, dtype=np.float64)
    fval = np.zeros(num_lines, dtype=np.float32)
    name = np.empty(num_lines, dtype=object)
    gamma = np.zeros(num_lines, dtype=np.float32) if linelist == 'atom' else None

    # Extract relevant data
    for i, line in enumerate(line_str):
        wavelist[i] = line['wrest']
        fval[i] = line['fval']
        name[i] = line['ion']
        if gamma is not None:
            gamma[i] = line['gamma']

    if method == 'Exact':
        match_idx = np.where(np.abs(lambda_rest - wavelist) < 1e-3)
    elif method == 'closest':
        match_idx = np.array([np.abs(lambda_rest - wavelist).argmin()])
    else:
        raise ValueError("Specify a valid matching method: 'closest' or 'Exact'")
    
    if gamma is not None:
        return {'wave': wavelist[match_idx], 'fval': fval[match_idx], 'name': name[match_idx], 'gamma': gamma[match_idx]}
    else:
        return {'wave': wavelist[match_idx], 'fval': fval[match_idx], 'name': name[match_idx]}

def read_line_list(label):
    """
    Reads and parses atomic line list data based on the specified label.

    Parameters:
    label (str): Specifies the line list ('atom', 'LLS', 'LLS Small', 'DLA').

    Returns:
    list of dict: Parsed atomic line data.
    """
    file_map = {
        'atom': 'lines/atom_full.dat',
        'LLS': 'lines/lls.lst',
        'LLS Small': 'lines/lls_sub.lst',
        'DLA': 'lines/dla.lst'
    }
    
    if label not in file_map:
        raise ValueError("Invalid line list label. Choose from 'atom', 'LLS', 'LLS Small', or 'DLA'")
    
    filename = resource_filename('rbvfit', file_map[label])
    data = []
    
    if label == 'atom':
        s = ascii.read(filename)
        for row in s:
            data.append({
                'wrest': float(row['col2']),
                'ion': f"{row['col1']} {int(row['col2'])}",
                'fval': float(row['col3']),
                'gamma': float(row['col4'])
            })
    else:
        with open(filename, 'r') as f:
            f.readline()  # Skip header
            for line in f:
                columns = line.strip().split()
                data.append({
                    'wrest': float(columns[0]),
                    'ion': f"{columns[1]} {columns[2]}",
                    'fval': float(columns[3])
                })
    
    return data
