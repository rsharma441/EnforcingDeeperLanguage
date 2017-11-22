import pandas as pd

def top_pct_split(mat, pct):
	new_mat=mat.sum(axis=1)/13 >=pct
	new_mat.index=mat.index
	return(new_mat)
