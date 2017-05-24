import EduNets, Base.vcat;

export EvaluationState, vcat;

type EvaluationState{A<:AbstractFloat}
	predicted::Vector{A};
	real::Vector{Int};
end

function EvaluationState(model::EduNets.AbstractModel, dataset::EduNets.AbstractDataset; forwardRuns::Int = 1)::EvaluationState
	predicted = zeros(AbstractFloat, length(dataset.y));
	for i in 1:forwardRuns
		predicted .+= EduNets.project!(model, dataset)[end, :];
	end
	if forwardRuns > 1
		predicted ./= forwardRuns;
	end
	return EvaluationState(predicted, dataset.y);
end

function vcat(states::EvaluationState...)::EvaluationState
	return EvaluationState(mapreduce(x->x.predicted, vcat, states), mapreduce(x->x.real, vcat, states));
end

