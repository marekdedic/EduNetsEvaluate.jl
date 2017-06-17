import EduNets.AbstractDataset, EduNets.AbstractModel, ThreadedMap.tmap;
import Plots;

export PRcurve, plotPRcurve, plotFscore;

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

function plotPRcurve(curves::Tuple{AbstractString, PRcurve}...)
	lims = (0, 1.05);
	Plots.plot(; xlabel = "Recall", ylabel = "Precision", xlims = lims, ylims = lims)
	pl(tuple) = Plots.plot!(tuple[2].recall, tuple[2].precision; label = tuple[1]);
	pl.(curves);
	Plots.gui();
end

plotPRcurve(curve::PRcurve) = plotPRcurve(("PR Curve", curve));

function plotFscore(curves::Tuple{AbstractString, PRcurve}...)
	Plots.plot(; xlabel = "Threshold", ylabel = "F1 score", ylims = (0, 1.05))
	function pl(tuple)
		fscore = 2 .* tuple[2].precision .* tuple[2].recall ./ (tuple[2].precision .+ tuple[2].recall);
		Plots.plot!(tuple[2].thresholds, fscore; label = tuple[1]);
	end
	pl.(curves);
	Plots.gui();
end

plotFscore(curve::PRcurve) = plotFscore(("F1 score", curve));

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
