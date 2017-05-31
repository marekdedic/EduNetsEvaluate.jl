export ROCcurveStage4;

type ROCcurveStage4{A<:AbstractFloat}
	thresholds::Vector{A};
	FP::Vector{Int}; # False positives
	RN::Int; # Real negatives
	TPR::Vector{A}; # True positive rate (= recall)
end

function ROCcurveStage4(S3::ROCcurveStage3)::ROCcurveStage4
	TPR::Vector{eltype(S3.thresholds)} = S3.TP ./ S3.RP;
	return ROCcurveStage4(S3.thresholds, S3.FP, S3.RN, TPR);
end

