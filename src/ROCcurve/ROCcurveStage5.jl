import Base.+, Base.vcat

export ROCcurveStage5, evaluate, +, vcat;

type ROCcurveStage5{A<:AbstractFloat}
	thresholds::Vector{A};
	FP::Vector{Int}; # False positives
	RN::Int; # Real negatives
end

function ROCcurveStage5(S4::ROCcurveStage4)::ROCcurveStage5
	thresholds = S4.thresholds;
	FP = zeros(Int, length(thresholds));
	return ROCcurveStage5(thresholds, FP, 0);
end

function evaluate(S5::ROCcurveStage5, state::EvaluationState)
	if size(state.predicted, 1) == 0
		return;
	end
	perm = sortperm(state.predicted);
	predicted = state.predicted[perm];
	real = state.real[perm];
	S5.RN = countnz(real .== 1);
	FPcounter = S5.RN;
	THcounter = 1;
	len = length(predicted);
	for i in 1:length(S5.thresholds)
		if predicted[1] < S5.thresholds[THcounter]
			break;
		end
		S5.FP[THcounter] = FPcounter;
		THcounter += 1;
	end
	if real[1] == 1
		FPcounter -= 1;
	end
	i = 1;
	while i <= (len - 1)
		if THcounter > length(S5.thresholds)
			break;
		end
		threshold = S5.thresholds[THcounter];
		if predicted[i] < threshold && predicted[i + 1] >= threshold
			S5.FP[THcounter] = FPcounter;
			THcounter += 1;
		else
			if real[i + 1] == 1
				FPcounter -= 1;
			end
			i += 1;
		end
	end
	S5.FP[THcounter:end] .= 0;
end

function ROCcurveStage5(S4::ROCcurveStage4, state::EvaluationState)::ROCcurveStage5
	S5 = ROCcurveStage5(S4);
	evaluate(S5, state);
	return S5;
end

function +(S5::ROCcurveStage5, state::EvaluationState)::ROCcurveStage5
	FP = deepcopy(S5.FP);
	RN = S5.RN
	evaluate(S5, state);
	S5.FP .+= FP;
	S5.RN += RN;
	return S5;
end

function vcat(stages::ROCcurveStage5...)::ROCcurveStage5
	FP = Vector{Int}(length(stages[1].thresholds))
	for i in 1:length(FP)
		FP[i] = mapreduce(x->x.FP[i], +, stages);
	end
	RN = mapreduce(x->x.RN, +, stages);
	return ROCcurveStage5(stages[1].thresholds, FP, RN);
end

