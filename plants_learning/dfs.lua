function dfs_rec(M, visited_nodes, node)
	if visited_nodes[node]==1 then
		return nil
	end
	if visited_nodes[node]==-1 then
		return {node}
	end
	visited_nodes[node] = 1
	local edges = M:sub(node,node)
	edges = edges:view(edges:nElement())

	local i = 1
	for i=1,edges:nElement() do
		if edges[i]==1 then
			local sol = dfs_rec(M, visited_nodes, i)
			if sol then
				sol[#sol+1] = node
				return sol
			end
		end
	end
	return nil
end

function get_L_rec(M, visited_nodes, node)
	if visited_nodes[node]==1 then
		return nil
	end
	visited_nodes[node] = 1
	local edges = M:sub(node,node)
	edges = edges:view(edges:nElement())
	local i = 1
	for i=1,edges:nElement() do
		if edges[i]==1 then
			get_L_rec(M, visited_nodes, i)
		end
	end
	return nil
end

function find_augmenting_path(M, G)
	local n_nodes = (#M)[1]
	local D = torch.zeros(n_nodes+1, n_nodes+1)
	local exposed_vertices = torch.ones(n_nodes) - torch.cmin(M:sum(2), 1)
	local exposed_vertices_a = exposed_vertices:clone()
	exposed_vertices_a:sub(1,n_nodes/2):fill(0)
	local exposed_vertices_b = exposed_vertices:clone()
	exposed_vertices_b:sub(n_nodes/2+1,n_nodes):fill(0)
	
	local small_G = G:sub(n_nodes/2+1,n_nodes, 1,n_nodes/2)
	local small_M = M:sub(n_nodes/2+1,n_nodes, 1,n_nodes/2)
	local G_minus_M = torch.cmax(small_G - small_M, 0)
	local G_and_M = torch.cmul(small_G, small_M)
	D[{{n_nodes/2+1,n_nodes},{1,n_nodes/2}}] = G_minus_M
	D[{{1,n_nodes/2},{n_nodes/2+1,n_nodes}}] = G_and_M:t()

	D[{{-1},{1,-2}}] = exposed_vertices_a

	local nodes = torch.zeros(1,n_nodes+1)
	nodes[{{}, {1,-2}}] = -exposed_vertices_b
	
	nodes = nodes:view(nodes:nElement())
	local dfs_result = dfs_rec(D, nodes, n_nodes+1)
	return dfs_result, D
end

function augmenting_path(oriG, initM)
	local n_nodes = (#oriG)[1]
	local G = torch.zeros(n_nodes*2, n_nodes*2)
	G[{{n_nodes+1, -1},{1,n_nodes}}] = oriG	
	G[{{1,n_nodes},{n_nodes+1, -1}}] = oriG:t()
	local M = initM or torch.zeros(#G)
	local P, C = find_augmenting_path(M, G)
	local counter = 0
	while P do		
		for i = 1,#P-2 do
			if M[P[i]][P[i+1]] == 0 then
				M[P[i]][P[i+1]] = 1
				M[P[i+1]][P[i]] = 1
			else
				M[P[i]][P[i+1]] = 0
				M[P[i+1]][P[i]] = 0
			end
		end
		counter = counter + 1
		P, D = find_augmenting_path(M, G)
	end

	nodes = torch.zeros(n_nodes*2+1)
	get_L_rec(D, nodes, n_nodes*2+1) 
	nodes = nodes:sub(1,-2)
	local nodes_A = torch.zeros(n_nodes*2)
	local nodes_B = torch.zeros(n_nodes*2)
	nodes_A:sub(n_nodes+1,n_nodes*2):fill(1)
	nodes_B:sub(1,n_nodes):fill(1)
	local C = torch.cmax(nodes_A-nodes, torch.zeros(n_nodes*2))
	C = C+torch.cmul(nodes, nodes_B) 

	local MM = M:sub(n_nodes+1, n_nodes*2, 1, n_nodes)
	if MM:sum()~=C:sum() then
		print '-----'
		print(MM:sum())
		print(C:sum())
		print '-----'
	end	
	return MM, C
end
