import Base.vcat

export PRcurveStage3, evaluate, vcat;

type PRcurveStage3{A<:AbstractFloat}
	thresholds::Vector{A};
	TP::Vector{Int}; # True Positives
	PP::Vector{Int}; # Predicted positives
	RP::Int; # Real Positives
end

function PRcurveStage3(S2::PRcurveStage2)::PRcurveStage3
	thresholds = S2.thresholds;
	TP = zeros(Int, length(thresholds));
	PP = zeros(Int, length(thresholds));
	return PRcurveStage3(thresholds, TP, PP, 0);
end

function evaluate(S3::PRcurveStage3, state::EvaluationState)
	if size(state.predicted, 1) == 0
		return;
	end
	perm = sortperm(state.predicted);
	predicted = state.predicted[perm];
	real = state.real[perm];
	S3.RP = countnz(real .== 2);
	TPcounter = S3.RP;
	THcounter = 1;
	for i in 1:length(S3.thresholds)
		if predicted[1] < S3.thresholds[THcounter]
			break;
		end
		S3.TP[THcounter] = TPcounter;
		S3.PP[THcounter] = length(predicted);
		THcounter += 1;
	end
	if(real[1] == 2)
		TPcounter -= 1;
	end
	i = 1;
	while i <= (length(predicted) - 1)
		if (predicted[i] <= S3.thresholds[THcounter] && predicted[i + 1] > S3.thresholds[THcounter])
			S3.TP[THcounter] = TPcounter;
			S3.PP[THcounter] = length(predicted[i + 1:end]);
			THcounter += 1;
		else
			if(real[i + 1] == 2)
				TPcounter -= 1;
			end
			i += 1;
		end
	end
end

function PRcurveStage3(S2::PRcurveStage2, state::EvaluationState)::PRcurveStage3
	S3 = PRcurveStage3(S2);
	evaluate(S3, state);
	return S3;
end

function vcat(stages::PRcurveStage3...)::PRcurveStage3
	nullclamp(vec::Vector, index::Int) = index <= length(vec) ? vec[index] : 0;

	len = length(stages[1].thresholds);
	TP = Vector{Int}(len);
	PP = Vector{Int}(len);
	RP = mapreduce(x->x.RP, +, stages);
	indices = ones(Int, length(stages));
	for i in 1:len
		TP[i] = mapreduce(x->nullclamp(x[2].TP, indices[x[1]]), +, enumerate(stages));
		PP[i] = mapreduce(x->nullclamp(x[2].PP, indices[x[1]]), +, enumerate(stages));
		validThresholds = Vector{eltype(stages[1].thresholds)}(length(stages))
		for j in 1:length(stages)
			if indices[j] <= length(stages[j].thresholds)
				validThresholds[j] = stages[j].thresholds[indices[j]];
			else
				validThresholds[j] = Inf;
			end
		end
		j = indmin(validThresholds);
		indices[j] += 1;
	end
	return PRcurveStage3(stages[1].thresholds, TP, PP, RP);
end

