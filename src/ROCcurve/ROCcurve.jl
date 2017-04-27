import Plots;
export ROCcurve, plot, plotROCcurve;

type ROCcurve
	TPR::Vector{AbstractFloat}; # True-positive rate
	FPR::Vector{AbstractFloat}; # False-positive rate
end

#=
function ROCcurve(length::Int)::ROCcurve
	return ROCcurve(zeros(length), zeros(length));
end
=#

#=
function ROCcurve(state::EvaluationState)::ROCcurve
	pmask = state.real .== 2; # Bool array, true when real == 2 i. e. for real positives
	nmask = state.real .== 1;
	curve = ROCcurve(sum(nmask))
	thresholds = sort(state.predicted[nmask], rev = true);
	for (i,it) in enumerate(thresholds)
		curve.FPR[i] = mean(state.predicted[nmask] .> it); # Mean over Bool array (percentage of true values), true when truly negative but prediction higher then threshold
		curve.TPR[i] = mean(state.predicted[pmask] .> it);
	end
	return curve;
end
=#

function ROCcurve(fragment::ROCcurveFragment)::ROCcurve
	TPR = fragment.TP ./ fragment.RP;
	FPR = fragment.FP ./ fragment.RN;
	return ROCcurve(TPR, FPR)
end

function ROCcurve(state::EvaluationState)
	return ROCcurve(ROCcurveFragment(state));
end

function ROCcurve(model::AbstractModel, dataset::AbstractDataset)
	return ROCcurve(ROCcurveFragment(model, dataset));
end

function plot(curve::ROCcurve)
	Plots.plot(curve.FPR, curve.TPR; xlabel = "False-positive rate", ylabel = "True-positive rate", xlims = (0, 1), ylims = (0, 1), label = "ROC Curve");
	Plots.plot!(identity; linestyle = :dot, label="");
end

function plot(fragment::ROCcurveFragment)
	plot(ROCcurve(fragment));
end

function plotROCcurve(state::EvaluationState)
	plot(ROCcurve(state));
end

function plotROCcurve(model::AbstractModel, dataset::AbstractDataset)
	plot(ROCcurve(model, dataset));
end

