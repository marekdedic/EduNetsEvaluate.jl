import Base.vcat

export ROCcurveStage3, evaluate, vcat;

type ROCcurveStage3{A<:AbstractFloat}
	thresholds::Vector{A};
	TP::Vector{Int}; # True Positives
	FP::Vector{Int}; # False positives
	RP::Int; # Real Positives
	RN::Int; # Real negatives
end

function ROCcurveStage3(S2::ROCcurveStage2)::ROCcurveStage3
	thresholds = S2.thresholds;
	len = length(thresholds);
	TP = Vector{Int}(len);
	FP = Vector{Int}(len);
	return ROCcurveStage3(thresholds, TP, FP, 0, 0);
end

function evaluate(S3::ROCcurveStage3, state::EvaluationState)
	if size(state.predicted, 1) == 0
		return;
	end
	perm = sortperm(state.predicted);
	predicted = state.predicted[perm];
	real = state.real[perm];
	S3.RP = countnz(real .== 2);
	S3.RN = countnz(real .== 1);
	TPcounter = S3.RP;
	FPcounter = S3.RN;
	THcounter = 1;
	len = length(predicted);
	for i in 1:length(S3.thresholds)
		if predicted[1] < S3.thresholds[THcounter]
			break;
		end
		S3.TP[THcounter] = TPcounter;
		S3.FP[THcounter] = FPcounter;
		THcounter += 1;
	end
	if(real[1] == 2)
		TPcounter -= 1;
	else
		FPcounter -= 1;
	end
	i = 1;
	while i <= (len - 1)
		if THcounter > length(S3.thresholds)
			break;
		end
		threshold = S3.thresholds[THcounter];
		if (predicted[i] < threshold && predicted[i + 1] >= threshold)
			S3.TP[THcounter] = TPcounter;
			S3.FP[THcounter] = FPcounter;
			THcounter += 1;
		else
			if(real[i + 1] == 2)
				TPcounter -= 1;
			else
				FPcounter -= 1;
			end
			i += 1;
		end
	end
	S3.TP[THcounter:end] .= 0;
	S3.FP[THcounter:end] .= 0;
end

function ROCcurveStage3(S2::ROCcurveStage2, state::EvaluationState)::ROCcurveStage3
	S3 = ROCcurveStage3(S2);
	evaluate(S3, state);
	return S3;
end

function vcat(stages::ROCcurveStage3...)::ROCcurveStage3
	len = length(stages[1].thresholds);
	TP = Vector{Int}(len);
	FP = Vector{Int}(len);
	for i in 1:length(FP)
		TP[i] = mapreduce(x->x.TP[i], +, stages);
		FP[i] = mapreduce(x->x.FP[i], +, stages);
	end
	RP = mapreduce(x->x.RP, +, stages);
	RN = mapreduce(x->x.RN, +, stages);
	return ROCcurveStage3(stages[1].thresholds, TP, FP, RP, RN);
end

