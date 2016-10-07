require 'dfs'

function Hung(B)
	local M = B:clone()
	M = M-M:min()
	local n = (#M)[1]
	local L = M:clone():fill(0)
	n_rows = (#M)[1]
	n_columns = (#M)[2]

	M:add(-torch.repeatTensor(torch.min(M,2), 1,n_columns))	
	M:add(-torch.repeatTensor(torch.min(M,1), n_rows,1))

	while true do
		local mask_M = torch.eq(M, torch.zeros(#M)):double()
		A, C = augmenting_path(mask_M)
		local counter = A:sum()
		if counter==n_rows then
			local assignments = {}
			for i=1,counter do
				for j=1,n_columns do
					if A[i][j]==1 then
						assignments[i] = j
					end
				end
			end
			return assignments
		end
		local marked_rows = C:sub(n+1,n*2)
		local marked_columns = C:sub(1,n)
		local rows_mask = (torch.ones(n)-marked_rows):view(n,1)
		local columns_mask = (torch.ones(n)-marked_columns):view(n,1)
		local mask = rows_mask*columns_mask:t()
		mask = mask:byte()
		local delta = M:maskedSelect(mask):min()
		M[mask] = M[mask] - delta

		rows_mask = marked_rows:view(n,1)
		columns_mask = marked_columns:view(1,n)
		local auxH = torch.repeatTensor(rows_mask,1 , n)
		local auxI = torch.repeatTensor(columns_mask, n, 1)

		local rubberedM = auxH + auxI
		M[torch.eq(rubberedM, torch.zeros(#rubberedM)+2)] = M[torch.eq(rubberedM, torch.zeros(#rubberedM)+2)] + delta
	end
	return assignments,M,L
end

