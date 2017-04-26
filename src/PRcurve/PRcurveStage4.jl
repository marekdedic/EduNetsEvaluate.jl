export PRcurveStage4;

type PRcurveStage4{A<:AbstractFloat}
	thresholds::Vector{A};
	TP::Vector{Int}; # True Positives
	PP::Vector{Int}; # Predicted positives
	recall::Vector{A};
end

function PRcurveStage4(S3::PRcurveStage3)::PRcurveStage4
	recall = S3.TP ./ S3.RP;
	return PRcurveStage4(S3.thresholds, S3.TP, S3.PP, recall);
end

