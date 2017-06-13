import EduNets.AbstractDataset, EduNets.AbstractModel, ThreadedMap.tmap;
import Plots;

export ROCcurve, plotROCcurve, plotFscore;

type ROCcurve{A<:AbstractFloat}
	thresholds::Vector{A};
	TPR::Vector{A}; # True positive rate
	FPR::Vector{A}; # False positive rate
end

function ROCcurve(S4::ROCcurveStage4)::ROCcurve
	FPR::Vector{eltype(S4.thresholds)} = S4.FP ./ S4.RN;
	return ROCcurve(S4.thresholds, S4.TPR, FPR);
end

function ROCcurve(S4::ROCcurveStage4, S5::ROCcurveStage5)::ROCcurve
	FPR::Vector{eltype(S4.thresholds)} = (S4.FP .+ S5.FP) ./ (S4.RN + S5.RN);
	return ROCcurve(S4.thresholds, S4.TPR, FPR);
end

# Plotting

function plotROCcurve(curve::ROCcurve)
	Plots.plot(curve.FPR, curve.TPR; xlabel = "False positive rate", ylabel = "True positive rate", xlims = (0, 1), ylims = (0, 1), label = "ROC Curve");
	Plots.plot!(identity; linestyle = :dot, label="");
end

# Complete ROCcurve functions

function ROCcurve(model::EduNets.AbstractModel, mixed::Vector{EduNets.AbstractDataset}, negative::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100, threaded::Bool = false)::ROCcurve
	fun = threaded ? ThreadedMap.tmap : map;
	statesMixed = fun(p->EvaluationState(model, p), mixed);
	statesNegative = fun(n->EvaluationState(model, n), negative);
	return ROCcurve(statesMixed, statesNegative; thresholdCount = thresholdCount);
end

function ROCcurve{T<:AbstractFloat}(mixed::Vector{EvaluationState{T}}, negative::Vector{EvaluationState{T}}; thresholdCount::Int = 100, threaded::Bool = false)::ROCcurve
	fun = threaded ? ThreadedMap.tmap : map;
	S1vec = fun(s->ROCcurveStage1(s), mixed);
	S1 = vcat(S1vec...);
	S2 = ROCcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = fun(s->ROCcurveStage3(S2, s), mixed);
	S3 = vcat(S3vec...);
	S4 = ROCcurveStage4(S3);
	S5vec = fun(n->ROCcurveStage5(S4, n), negative);
	S5 = vcat(S5vec...);
	return ROCcurve(S4, S5);
end

function ROCcurve(model::EduNets.AbstractModel, mixed::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100, threaded::Bool = false)::ROCcurve
	fun = threaded ? ThreadedMap.tmap : map;
	statesMixed = fun(p->EvaluationState(model, p), mixed);
	return ROCcurve(statesMixed; thresholdCount = thresholdCount);
end

function ROCcurve{T<:AbstractFloat}(mixed::Vector{EvaluationState{T}}; thresholdCount::Int = 100, threaded::Bool = false)::ROCcurve
	fun = threaded ? ThreadedMap.tmap : map;
	S1vec = fun(s->ROCcurveStage1(s), mixed);
	S1 = vcat(S1vec...);
	S2 = ROCcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = fun(s->ROCcurveStage3(S2, s), mixed);
	S3 = vcat(S3vec...);
	S4 = ROCcurveStage4(S3);
	return ROCcurve(S4);
end
