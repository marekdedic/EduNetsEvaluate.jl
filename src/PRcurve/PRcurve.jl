import EduNets.AbstractDataset, EduNets.AbstractModel, ThreadedMap.tmap;
import Plots;

export PRcurve, plotPRcurve, plotFBetaScore, plotFscore;

type PRcurve{A<:AbstractFloat}
	thresholds::Vector{A};
	precision::Vector{A};
	recall::Vector{A};
end

function PRcurve(S4::PRcurveStage4)::PRcurve
	precision::Vector{eltype(S4.thresholds)} = S4.TP ./ S4.PP;
	precision[S4.PP .== 0] .= 1;
	return PRcurve(S4.thresholds, precision, S4.recall);
end

function PRcurve(S4::PRcurveStage4, S5::PRcurveStage5)::PRcurve
	PP = S4.PP .+ S5.PP;
	precision::Vector{eltype(S4.thresholds)} = S4.TP ./ PP;
	precision[PP .== 0] .= 1;
	return PRcurve(S4.thresholds, precision, S4.recall);
end

# Plotting

function plotPRcurve{S<:AbstractString, T<:AbstractFloat}(labels::Vector{S}, curves::Vector{PRcurve{T}}; title::AbstractString = "")
	lims = (0, 1.05);
	Plots.plot(; xlabel = "Recall", ylabel = "Precision", xlims = lims, ylims = lims, title = title)
	pl(label, curve) = Plots.plot!(curve.recall, curve.precision; label = label);
	pl.(labels, curves);
	Plots.gui();
end

plotPRcurve(curve::PRcurve; title::AbstractString = "") = plotPRcurve(["PR Curve"], [curve]; title = title);

function plotFscore{S<:AbstractString, T<:AbstractFloat}(labels::Vector{S}, curves::Vector{PRcurve{T}}; beta::AbstractFloat = 1.0, title::AbstractString = "")
	Plots.plot(; xlabel = "Threshold", ylabel = "F" * string(beta) * " score", ylims = (0, 1.05), title = title)
	function pl(label, curve)
		fscore = (1 + beta^2) .* curve.precision .* curve.recall ./ ((beta^2 * curve.precision) .+ curve.recall);
		Plots.plot!(curve.thresholds, fscore; label = label);
	end
	pl.(labels, curves);
	Plots.gui();
end

plotFscore(curve::PRcurve; beta::AbstractFloat = 1.0, title::AbstractString = "") = plotFscore(["F" * string(beta) * " score"], [curve]; beta = beta, title = title);

# Complete PRcurve functions

function PRcurve(model::EduNets.AbstractModel, mixed::Vector{EduNets.AbstractDataset}, negative::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100, threaded::Bool = false)::PRcurve
	fun = threaded ? ThreadedMap.tmap : map;
	statesMixed = fun(p->EvaluationState(model, p), mixed);
	statesNegative = fun(n->EvaluationState(model, n), negative);
	return PRcurve(statesMixed, statesNegative; thresholdCount = thresholdCount);
end

function PRcurve{T<:AbstractFloat}(mixed::Vector{EvaluationState{T}}, negative::Vector{EvaluationState{T}}; thresholdCount::Int = 100, threaded::Bool = false)::PRcurve
	fun = threaded ? ThreadedMap.tmap : map;
	S1vec = fun(s->PRcurveStage1(s), mixed);
	S1 = vcat(S1vec...);
	S2 = PRcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = fun(s->PRcurveStage3(S2, s), mixed);
	S3 = vcat(S3vec...);
	S4 = PRcurveStage4(S3);
	S5vec = fun(n->PRcurveStage5(S4, n), negative);
	S5 = vcat(S5vec...);
	return PRcurve(S4, S5);
end

function PRcurve(model::EduNets.AbstractModel, mixed::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100, threaded::Bool = false)::PRcurve
	fun = threaded ? ThreadedMap.tmap : map;
	statesMixed = fun(p->EvaluationState(model, p), mixed);
	return PRcurve(statesMixed; thresholdCount = thresholdCount);
end

function PRcurve{T<:AbstractFloat}(mixed::Vector{EvaluationState{T}}; thresholdCount::Int = 100, threaded::Bool = false)::PRcurve
	fun = threaded ? ThreadedMap.tmap : map;
	S1vec = fun(s->PRcurveStage1(s), mixed);
	S1 = vcat(S1vec...);
	S2 = PRcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = fun(s->PRcurveStage3(S2, s), mixed);
	S3 = vcat(S3vec...);
	S4 = PRcurveStage4(S3);
	return PRcurve(S4);
end
