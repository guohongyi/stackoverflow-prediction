#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:26:58 2020

@author: Xue
"""
import csv
import numpy as np
#Read data
def read_csv(file_name, my_delim=',', my_quote='"'):
    len_csv = 0
    file_content = []
    with open(file_name, 'rU') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=my_delim, quotechar=my_quote)
        
        counter = 0
        for row in spamreader:
            #print ', '.join(row)
            if len_csv == 0:
                len_csv = len(row)
            elif len_csv != len(row):
                print('[warning] row %i size not equal %i/%i' % (counter, len(row), len_csv))
                print(', '.join(row))
            
            clean_row = []
            for list_item in row:
                if len(list_item) > 0 and list_item[-1] in [' ']:
                    list_item = list_item[:-1]
                if ',' in list_item:
                    list_item=list_item.replace(',',' ')
                #list_item=float(list_item)
                clean_row.append(list_item)
            file_content.append(clean_row)
            counter += 1        
        csv_np = np.array(file_content)    
    #file_contnet_pd = pd.read_csv(file_path)
    return csv_np

predicted_tag = ['python','r']
data_path = './dataExpert.csv'
experts = read_csv(data_path)
predicted_experts = np.empty([0])
for i1 in range(len(predicted_tag)):
    idx = np.where(experts[:,0]==predicted_tag[i1])[0]
    predicted_experts = np.append(predicted_experts,experts[idx,1:])
unique_experts, ind = np.unique(predicted_experts, return_inverse = True)
predicted_experts = ", ".join(unique_experts.tolist())
print(predicted_experts)
# =============================================================================
# expert=	{
#     '.net'	: 'Jon Skeet, Gordon Linoff',
#     'andorid'	: 'CommonsWare,	Gordon Linoff',	
# 	'arrays'	: 'Barmar, Gordon Linoff',	
# 	'asp.net'	: 'Jon Skeet, Gordon Linoff',	
# 	'c#	:' : 'Jon Skeet, Gordon Linoff',	
# 	'c++'	: 'Barmar, Martijin Pieters',	
# 	'css'	: 'Barmar, VonC',	
# 	'html'	: 'Barmar, VonC',	
# 	'ios'	: 'VonC, MadProgrammer',	
# 	'java'	: 'MadProgrammer, Gordon Linoff	',
# 	'javascript'	: 'Barmar, Gordon Linoff',	
# 	'jquery'	: 'Barmar, Gordon Linoff',	
# 	'mysql'	: '	Gordon Linoff, Barmar',	
# 	'node.js'	: 'VonC, Barmar',	
# 	'php'	: 'Barmar, Barmar',
# 	'python' : 'Martijn Pieters, Barmar',	
# 	'r'	: 'Gordon Linoff, VonC',
# 	'ruby-on-rails'	: '	VonC, Barmar',	
# 	'sql' : 'Gordon Linoff, Barmar'}
# =============================================================================
