import Plots;

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

