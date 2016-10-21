require 'torch'
local table_tensors = {}
function table_tensors.table2vector (table)
	if torch.isTensor(table) then
		T = table:clone():resize(table:nElement())
		return T
	end

	local n_numbers = 0
	for k,v in ipairs(table) do
		n_numbers = n_numbers + v:nElement()
	end
	local T = torch.Tensor(n_numbers)
	local pointer = 1
	for k,v in ipairs(table) do
	--	local aux = v:clone():resize(v:nElement())
		T[{{pointer, pointer+v:nElement()-1}}] = v:resize(v:nElement())
		pointer = pointer+v:nElement()
	end
	return T
end

function table_tensors.vector2table (T, table)
	if torch.isTensor(table) then
		table_view = table:view(table:nElement())
		for i =1,table:nElement() do
			table_view[i] = T[i]
		end
		return
	end

	local pointer = 1
	for k,v in ipairs(table) do
		table[k] = T:sub(pointer, pointer+v:nElement()-1):clone():reshape(#v)
		pointer = pointer+v:nElement()
	end
end

return table_tensors
