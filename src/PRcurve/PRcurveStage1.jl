export PRcurveStage1, vcat;

type PRcurveStage1{A<:AbstractFloat}
	thresholds::Vector{A}
end

function PRcurveStage1(state::EvaluationState)::PRcurveStage1
	return PRcurveStage1(state.predicted);
end

function vcat(stages::PRcurveStage1...)::PRcurveStage1
	return PRcurveStage1(mapreduce(x->x.thresholds, vcat, stages));
end

