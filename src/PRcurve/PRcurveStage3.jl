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
	TP = Vector{Int}(size(thresholds, 1));
	PP = Vector{Int}(size(thresholds, 1));
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
	len = size(predicted, 1);
	for i in 1:size(S3.thresholds, 1)
		if predicted[1] < S3.thresholds[THcounter]
			break;
		end
		S3.TP[THcounter] = TPcounter;
		S3.PP[THcounter] = len;
		THcounter += 1;
	end
	if real[1] == 2
		TPcounter -= 1;
	end
	i = 1;
	while i <= (len - 1)
		if THcounter > size(S3.thresholds, 1)
			break;
		end
		threshold = S3.thresholds[THcounter];
		if (predicted[i] < threshold && predicted[i + 1] >= threshold)
			S3.TP[THcounter] = TPcounter;
			S3.PP[THcounter] = len - i;
			THcounter += 1;
		else
			if real[i + 1] == 2
				TPcounter -= 1;
			end
			i += 1;
		end
	end
	S3.TP[THcounter:end] .= 0;
	S3.PP[THcounter:end] .= 0;
end

function PRcurveStage3(S2::PRcurveStage2, state::EvaluationState)::PRcurveStage3
	S3 = PRcurveStage3(S2);
	evaluate(S3, state);
	return S3;
end

function vcat(stages::PRcurveStage3...)::PRcurveStage3
	len = size(stages[1].thresholds, 1);
	TP = Vector{Int}(len);
	PP = Vector{Int}(len);
	for i in 1:len
		TP[i] = mapreduce(x->x.TP[i], +, stages);
		PP[i] = mapreduce(x->x.PP[i], +, stages);
	end
	RP = mapreduce(x->x.RP, +, stages);
	return PRcurveStage3(stages[1].thresholds, TP, PP, RP);
end

