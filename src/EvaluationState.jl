import EduNets, Base.vcat;

export EvaluationState, vcat;

type EvaluationState{A<:AbstractFloat}
	predicted::Vector{A};
	real::Vector{Int};
end

function EvaluationState(model::EduNets.AbstractModel, dataset::EduNets.AbstractDataset; forwardRuns::Int = 1)::EvaluationState
	predicted = zeros(AbstractFloat, length(dataset.y));
	for i in 1:forwardRuns
		predicted .+= forward!(model, dataset)[end][end, :];
	end
	predicted ./= forwardRuns;
	return EvaluationState(predicted, dataset.y);
end

function vcat(s1::EvaluationState, s2::EvaluationState)::EvaluationState
	return EvaluationState(vcat(s1.predicted, s2.predicted), vcat(s1.real, s2.real));
end

