export ROCcurveFragment, vcat;

type ROCcurveFragment
	thresholds::Vector{AbstractFloat};
	TP::Vector{AbstractFloat} # True Positives
	FP::Vector{AbstractFloat} # False positives
	RP::Int # Real Positives
	RN::Int # Real Negatives
end

function ROCcurveFragment(state::EvaluationState)::ROCcurveFragment
	perm = sortperm(state.predicted);
	thresholds = state.predicted[perm];
	real = state.real[perm];
	RP = countnz(state.real .== 2);
	RN = countnz(state.real .== 1);
	TP = Vector{AbstractFloat}(length(thresholds));
	FP = Vector{AbstractFloat}(length(thresholds));
	TPcounter = countnz(real .== 2);
	FPcounter = countnz(real .== 1);
	for i in 1:length(thresholds)
		TP[i] = TPcounter;
		FP[i] = FPcounter;
		if(real[i] == 2)
			TPcounter -= 1;
		else
			FPcounter -= 1;
		end
	end
	return ROCcurveFragment(thresholds, TP, FP, RP, RN);
end

function ROCcurveFragment(model::AbstractModel, dataset::AbstractDataset)
	return ROCcurveFragment(EvaluationState(model, dataset));
end

function vcat(fragments::ROCcurveFragment...)::ROCcurveFragment
	nullclamp(vec::Vector{AbstractFloat}, index::Int) = index <= length(vec) ? vec[index] : 0;

	len = mapreduce(x->length(x.thresholds), +, fragments);
	thresholds = Vector{AbstractFloat}(len);
	TP = Vector{AbstractFloat}(len);
	FP = Vector{AbstractFloat}(len);
	RP = mapreduce(x->x.RP, +, fragments);
	RN = mapreduce(x->x.RN, +, fragments);
	indices = ones(Int, length(fragments));
	for i in 1:len
		TP[i] = mapreduce(x->nullclamp(x[2].TP, indices[x[1]]), +, enumerate(fragments));
		FP[i] = mapreduce(x->nullclamp(x[2].FP, indices[x[1]]), +, enumerate(fragments));
		validThresholds = Vector{AbstractFloat}(length(fragments))
		for j in 1:length(fragments)
			if indices[j] <= length(fragments[j].thresholds)
				validThresholds[j] = fragments[j].thresholds[indices[j]];
			else
				validThresholds[j] = Inf;
			end
		end
		j = indmin(validThresholds);
		thresholds[i] = fragments[j].thresholds[indices[j]];
		indices[j] += 1;
	end
	return ROCcurveFragment(thresholds, TP, FP, RP, RN);
end

