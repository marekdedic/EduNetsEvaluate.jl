import Plots, EduNets.AbstractDataset, EduNets.AbstractModel;

export PRcurve, plotPRcurve, plotFscore;

type PRcurve{A<:AbstractFloat}
	thresholds::Vector{A};
	precision::Vector{A};
	recall::Vector{A};
end

function PRcurve(S4::PRcurveStage4)::PRcurve
	precision = S4.TP ./ S4.PP;
	return PRcurve(S4.thresholds, precision, S4.recall);
end

function PRcurve(S4::PRcurveStage4, S5::PRcurveStage5)::PRcurve
	precision = S4.TP ./ (S4.PP .+ S5.PP);
	return PRcurve(S4.thresholds, precision, S4.recall);
end

function plotPRcurve(curve::PRcurve)
	Plots.plot(curve.recall, curve.precision; xlabel = "Recall", ylabel = "Precision", xlims = (0, 1.05), ylims = (0, 1.05), label = "PR Curve");
end

function plotFscore(curve::PRcurve)
	fscore = 2 .* curve.precision .* curve.recall ./ (curve.precision .+ curve.recall);
	Plots.plot(curve.thresholds, fscore; xlabel = "Threshold", ylabel = "F1 score", ylims = (0, 1.05), label = "F1 score");
end

# Complete PRcurve functions

function PRcurve(model::EduNets.AbstractModel, positive::Vector{EduNets.AbstractDataset}, negative::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100)::PRcurve
	function processS5(model::EduNets.AbstractModel, dataset::EduNets.AbstractDataset, S4::PRcurveStage4)::PRcurveStage5
		state = EvaluationState(model, dataset);
		return PRcurveStage5(S4, state);
	end

	statesPositive = map(p->EvaluationState(model, p), positive);
	S1vec = map(s->PRcurveStage1(s), statesPositive);
	S1 = vcat(S1vec...);
	S2 = PRcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = map(s->PRcurveStage3(S2, s), statesPositive);
	S3 = vcat(S3vec...);
	S4 = PRcurveStage4(S3);
	S5vec = map(n->processS5(model, n, S4), negative);
	S5 = vcat(S5vec...);
	return PRcurve(S4, S5);
end

function PRcurve(model::EduNets.AbstractModel, positive::Vector{EduNets.AbstractDataset}; thresholdCount::Int = 100)::PRcurve
	statesPositive = map(p->EvaluationState(model, p), positive);
	S1vec = map(s->PRcurveStage1(s), statesPositive);
	S1 = vcat(S1vec...);
	S2 = PRcurveStage2(S1, thresholdCount = thresholdCount);
	S3vec = map(s->PRcurveStage3(S2, s), statesPositive);
	S3 = vcat(S3vec...);
	S4 = PRcurveStage4(S3);
	return PRcurve(S4);
end

function PRcurve(model::EduNets.AbstractModel, dataset::EduNets.AbstractDataset; thresholdCount::Int = 100)::PRcurve
	return PRcurve(model, [dataset]; thresholdCount = thresholdCount);
end
