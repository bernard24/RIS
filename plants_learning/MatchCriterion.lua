local MatchCriterion, parent = torch.class('nn.MatchCriterion', 'nn.Criterion')

function MatchCriterion:__init(lambda)
   parent.__init(self)
   self.lambda = lambda
   self.gradInput = {}

   self.criterion = nn.BCECriterion()
   self.gt_class = torch.Tensor(1)
   self.assignments = {}
   self.who_min_ss = {}
   if gpumode==1 then
	self.criterion:cuda()
	self.gt_class = self.gt_class:cuda()
   end
end

function iou(x,y)
	a = x:clone()
	b = y:clone()
	local iou_inter = a:clone():cmul(b):sum()
	return iou_inter/(a:sum()+b:sum()-iou_inter)
end

function MatchCriterion:updateOutput(qs_and_ss, ys)
   require 'Hung'
   local qs = qs_and_ss[1]:clone()
   local ss = qs_and_ss[2]:clone()

   local original_dimensionality = (#qs)[1]
   local elements_prediction = (#qs)[2]
   local elements_gt = 0
   self.assignments = {}
   if ys:nElement()>0 then
	elements_gt = (#ys)[2]
   end
   local zero = torch.Tensor(1):fill(0)
   local one = torch.Tensor(1):fill(1)

   local M_size = elements_gt
   local M = torch.Tensor(M_size, M_size):fill(0)
   if gpumode==1 then
	zero = zero:cuda()
	one = one:cuda()
   end
   for i = 1, math.min(elements_prediction,elements_gt) do
	   for j = 1, elements_gt do
			M[i][j] = -iou(qs:sub(1,-1,i,i), ys:sub(1,-1,j,j))
			M[i][j] = M[i][j] + self.lambda*self.criterion:forward(ss:sub(i,i), one)
       end
   end
   if elements_gt>0 then
	   self.assignments = Hung(M)
   end
   self.output = 0

if not self.assignments then
	torch.save('M.t7', M)
end

   for i = 1, elements_gt do --#self.assignments do
	self.output = self.output + M[i][self.assignments[i]]
   end
   for i = elements_gt+1, elements_prediction do
   	self.output = self.output + self.lambda*self.criterion:forward(ss:sub(i,i), zero)
   end
   return self.output
end

function MatchCriterion:updateGradInput(qs_and_ss, ys)
   local qs = qs_and_ss[1]:clone()
   local ss = qs_and_ss[2]:clone()
   local grad_qs = qs:clone():fill(0)
   local grad_ss = ss:clone():fill(0)

   local original_dimensionality = (#qs)[1]
   local elements_prediction = (#qs)[2]
   local elements_gt = 0
   if ys:nElement()>0 then
	elements_gt = (#ys)[2]
   end   

   local zero = torch.Tensor(1):fill(0)
   local one = torch.Tensor(1):fill(1)

   if gpumode==1 then
        zero = zero:cuda()
        one = one:cuda()
   end
   for i = 1, elements_prediction do --#self.assignments do
	if i<=elements_gt then
		local q = qs:sub(1,-1,i,i)
		local y = ys:sub(1,-1,self.assignments[i],self.assignments[i])

		local num = q:t()*y
		num:resize(num:nElement())
		local den =  - num
		local aux2 = q:t()*y
		aux2 = aux2[1][1]
		local aux = torch.sum(q) + torch.sum(y) - aux2
		local den = torch.Tensor(1):fill(aux)
		if gpumode==1 then
			den = den:cuda()
		end
		num = num[torch.LongStorage{1}]
		den = den[torch.LongStorage{1}]

		aux_den = torch.Tensor(#y):fill(den)
		aux_ones = torch.Tensor(#y):fill(1)
		aux_num = torch.Tensor(#y):fill(num)
		if gpumode==1 then
			aux_den = aux_den:cuda()
			aux_ones = aux_ones:cuda()
			aux_num = aux_num:cuda()
		end

		local aux1 = torch.cmul(aux_den,y)
		local aux = -(aux1 - torch.cmul(aux_ones-y, aux_num))

		aux_den2 = torch.Tensor(#aux):fill(den^2)
		if gpumode==1 then
			aux_den2 = aux_den2:cuda()
		end

		local aux = torch.cdiv(aux, aux_den2)

		grad_qs[{{},{i}}] = aux:resize(aux:nElement())
		grad_ss[i] = self.criterion:backward(ss:sub(i,i), one)*self.lambda
	else		
		grad_ss[i] = self.criterion:backward(ss:sub(i,i), zero)*self.lambda
	end
   end
   self.gradInput[1] = grad_qs
   self.gradInput[2] = grad_ss
   return self.gradInput
end
