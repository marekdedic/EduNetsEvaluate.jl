export ROCcurveStage1, vcat;

type ROCcurveStage1{A<:AbstractFloat}
	thresholds::Vector{A}
end

function ROCcurveStage1(state::EvaluationState)::ROCcurveStage1
	return ROCcurveStage1(state.predicted);
end

function vcat(stages::ROCcurveStage1...)::ROCcurveStage1
	return ROCcurveStage1(mapreduce(x->x.thresholds, vcat, stages));
end

