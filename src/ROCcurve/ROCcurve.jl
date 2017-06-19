import EduNets.AbstractDataset, EduNets.AbstractModel, ThreadedMap.tmap;
import Plots;

export ROCcurve, plotROCcurve, plotDETcurve;

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

function plotROCcurve{S<:AbstractString, T<:AbstractFloat}(labels::Vector{S}, curves::Vector{ROCcurve{T}}; log::Bool = false, title::AbstractString = "")
	lims = (0, 1);
	if log
		Plots.plot(0.001:0.001:1, 0.001:0.001:1; linestyle= :dot, label = "", xlabel = "False positive rate", ylabel = "True positive rate", xlims = (0.001, 1), ylims = lims, xscale = :log10, title = title);
	else
		Plots.plot(identity; linestyle= :dot, label = "", xlabel = "False positive rate", ylabel = "True positive rate", xlims = lims, ylims = lims, title = title);
	end
	pl(label, curve) = Plots.plot!(curve.FPR, curve.TPR; label = label);
	pl.(labels, curves);
	Plots.gui();
end

plotROCcurve(curve::ROCcurve; log::Bool = false, title::AbstractString = "") = plotROCcurve(["ROC Curve"], [curve]; log = log, title = title);

function plotDETcurve{S<:AbstractString, T<:AbstractFloat}(labels::Vector{S}, curves::Vector{ROCcurve{T}}; title::AbstractString = "")
	qnorm(x) = sqrt(2) * erfinv(2x - 1);
	lims = (qnorm(0.001), qnorm(0.55));
	tickvalues = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 50];
	ticks = (qnorm.(tickvalues ./ 100), string.(tickvalues));
	Plots.plot(; xlabel = "False positive rate (%)", ylabel = "False negative rate (%)", xlims = lims, ylims = lims, xticks = ticks, yticks = ticks, aspect_ratio = :equal, title = title);
	pl(label, curve) =	Plots.plot!(qnorm.(curve.FPR), qnorm.(1 .- curve.TPR); label = label);
	pl.(labels, curves);
	Plots.gui();
end

plotDETcurve(curve::ROCcurve; title::AbstractString = "") = plotDETcurve(["DET Curve"], [curve]; title = title);

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
