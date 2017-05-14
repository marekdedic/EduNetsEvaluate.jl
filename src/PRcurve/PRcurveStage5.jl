import Base.+, Base.vcat

export PRcurveStage5, evaluate, +, vcat;

type PRcurveStage5{A<:AbstractFloat}
	thresholds::Vector{A};
	PP::Vector{Int}; # Predicted positives
end

function PRcurveStage5(S4::PRcurveStage4)::PRcurveStage5
	thresholds = S4.thresholds;
	PP = zeros(Int, length(thresholds));
	return PRcurveStage5(thresholds, PP);
end

function evaluate(S5::PRcurveStage5, state::EvaluationState)
	if size(state.predicted, 1) == 0
		return;
	end
	predicted = sort(state.predicted);
	THcounter = 1;
	len = length(predicted);
	for i in 1:length(S5.thresholds)
		if predicted[1] < S5.thresholds[THcounter]
			break;
		end
		S5.PP[THcounter] = len;
		THcounter += 1;
	end
	i = 1;
	while i <= (len - 1)
		if THcounter > length(S5.thresholds)
			break;
		end
		if predicted[i] < S5.thresholds[THcounter] && predicted[i + 1] >= S5.thresholds[THcounter]
			S5.PP[THcounter] = len - i;
			THcounter += 1;
		else
			i += 1;
		end
	end
	S5.PP[THcounter:end] .= 0;
end

function PRcurveStage5(S4::PRcurveStage4, state::EvaluationState)::PRcurveStage5
	S5 = PRcurveStage5(S4);
	evaluate(S5, state);
	return S5;
end

function +(S5::PRcurveStage5, state::EvaluationState)::PRcurveStage5
	PP = deepcopy(S5.PP);
	evaluate(S5, state);
	S5.PP .+= PP;
	return S5;
end

function vcat(stages::PRcurveStage5...)::PRcurveStage5
	PP = Vector{Int}(length(stages[1].thresholds))
	for i in 1:length(PP)
		PP[i] = mapreduce(x->x.PP[i], +, stages);
	end
	return PRcurveStage5(stages[1].thresholds, PP);
end

